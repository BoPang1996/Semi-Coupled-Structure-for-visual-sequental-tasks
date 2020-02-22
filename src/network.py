import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from sync_batchnorm import SynchronizedBatchNorm2d
import random


class SCS(nn.Module):
    def __init__(self, batch_norm, num_action, dropout, q, test_scheme=1, img_size=112, syn_bn=False):

        super(SCS, self).__init__()

        # Configs
        self.channels = [3, 64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        self.RNN_layer = len(self.channels) - 1
        self.input_size = [img_size, img_size/4, img_size/4, img_size/4, img_size/4, img_size/4, img_size/8, img_size/8, img_size/8, img_size/8, img_size/16, img_size/16, img_size/16, img_size/16, img_size/32, img_size/32, img_size/32]
        self.out_size = [img_size/4, img_size/4, img_size/4, img_size/4, img_size/4, img_size/8, img_size/8, img_size/8, img_size/8, img_size/16, img_size/16, img_size/16, img_size/16, img_size/32, img_size/32, img_size/32, img_size/32]
        self.stride = [2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
        self.shortCutLayer =   [   1, 3, 5, 7, 9,  11, 13, 15]
        self.mergeShortCutLayer = [2, 4, 6, 8, 10, 12, 14, 16]
        self.not_selftrans = [6, 10, 14]

        # Modules
        self.dropout_RNN = nn.Dropout2d(p=dropout[0])

        self.RNN = nn.ModuleList([self.make_SCSCell((self.channels[i]), (self.channels[i + 1] + self.channels[i]), self.channels[i + 1], q=q, kernel_size=7 if i == 0 else 3, stride=self.stride[i], syn_bn=syn_bn, pool=(i == 0), padding=3 if i == 0 else 1) for i in range(self.RNN_layer)])
        self.init_weight(self.RNN, xavier_gain=3.0)

        self.ofm = nn.ModuleList([OFM(64, 16), OFM(128, 32), OFM(256, 32), OFM(512, 32)])

        block = [[{'convsc_1': [self.channels[5], self.channels[7], 1, 2, 0]}], [{'convsc_2': [self.channels[9], self.channels[11], 1, 2, 0]}],
                 [{'convsc_3': [self.channels[13], self.channels[15], 1, 2, 0]}]]
        self.ShortCut = nn.ModuleList([self._make_layer(block[i], batch_norm, syn_bn=syn_bn) for i in range(len(block))])
        self.init_weight(self.ShortCut)

        self.classifier = nn.Sequential(nn.Linear(int(self.channels[15] * pow(self.out_size[-1],2)), int(self.channels[15]*self.out_size[-1])), nn.ReLU(inplace=True), nn.Dropout(p=dropout[1]), nn.Linear(int(self.channels[15]*self.out_size[-1]), num_action))#nn.Softmax(dim=1))
        self.sftmx = nn.LogSoftmax(dim=1)
        self.init_weight(self.classifier)

        # Parameters
        self.syn_bn = syn_bn
        self.num_action = num_action
        self.test_scheme = test_scheme

    def init_weight(self, model, xavier_gain=1.0):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=xavier_gain)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, net_dict, batch_norm=False, syn_bn=False):
        layers = []
        length = len(net_dict)
        for i in range(length):
            one_layer = net_dict[i]
            key = list(one_layer.keys())[0]
            v = one_layer[key]

            if 'pool' in key:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                if batch_norm:
                    if syn_bn:
                        layers += [conv2d, SynchronizedBatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def make_SCSCell(self, in_channel1, in_channel2, out_channel, q, kernel_size, stride, padding, pool=False, syn_bn=False):
        class scs_cell(nn.Module):
            def __init__(self, in_channel1, in_channel2, out_channel, pool, q, kernel_size, stride, padding):
                super(scs_cell, self).__init__()
                self.outchannel = out_channel
                conv_data = nn.Conv2d(in_channels=in_channel1, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
                conv_ctrl = nn.Conv2d(in_channels=in_channel2, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=True)
                self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

                if syn_bn:
                    layers_data = [conv_data, SynchronizedBatchNorm2d(out_channel), torch.nn.ReLU()]
                    layers_ctrl = [conv_ctrl, SynchronizedBatchNorm2d(out_channel), torch.nn.Sigmoid()]
                else:
                    layers_data = [conv_data, nn.BatchNorm2d(out_channel), torch.nn.ReLU()]
                    layers_ctrl = [conv_ctrl, nn.BatchNorm2d(out_channel)]

                self.conv_data = nn.Sequential(*layers_data)
                self.conv_ctrl = nn.Sequential(*layers_ctrl)

                self.ispool = pool
                self.q = q
                self.stride = stride

            def forward(self, x, c, single_data, single_temp, single_c):
                input_data = x
                rand1 = random.random()
                rand2 = random.random()
                if rand1 < self.q:
                    x_ctrl = x.detach()
                else:
                    x_ctrl = x

                if rand2 < (1 - self.q):
                    input_data = x.detach()

                if rand2 < (1 - self.q) and rand1 < self.q:
                    if self.q < 0.5:
                        x_ctrl = x
                    else:
                        input_data = x

                input_ctrl = torch.cat((x_ctrl, c), 1)

                # main stream calculate
                data = self.conv_data(input_data)
                ctrl = torch.sigmoid(self.conv_ctrl(input_ctrl))

                if self.stride == 1:
                    output = data * ctrl
                else:
                    output = data * self.pool(ctrl)

                # data stream calculate
                data_l = self.conv_data(single_data)

                # temp stream calculate
                input_temp = torch.cat((single_temp, single_c), 1)
                temp = self.conv_ctrl(input_temp)
                temp_t = torch.sigmoid(temp)
                if self.stride == 1:
                    temp_l = F.relu(temp)
                else:
                    temp_l = self.pool(F.relu(temp))

                return output, ctrl, data, data_l, temp_t, temp_l

        return scs_cell(in_channel1, in_channel2, out_channel, pool, q, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, initial, initial_single_temp):
        out_action = torch.zeros((x.shape[0], x.shape[1], self.num_action)).cuda().float()
        data_action = torch.zeros((x.shape[0], x.shape[1], self.num_action)).cuda().float()
        temp_map_4 = torch.zeros((x.shape[0], x.shape[1], 2, x.shape[3] // 4, x.shape[3] // 4)).cuda().float()
        temp_map_8 = torch.zeros((x.shape[0], x.shape[1], 2, x.shape[3] // 8, x.shape[3] // 8)).cuda().float()
        temp_map_12 = torch.zeros((x.shape[0], x.shape[1], 2, x.shape[3] // 16, x.shape[3] // 16)).cuda().float()
        temp_map_16 = torch.zeros((x.shape[0], x.shape[1], 2, x.shape[3] // 32, x.shape[3] // 32)).cuda().float()
        for frame in range(x.shape[1]):

            # 0
            out, initial[0], data_map, data, initial_single_temp[0], temp = self.RNN[0](x[:, frame], initial[0],
                                                                                        x[:, frame], x[:, frame],
                                                                                        initial_single_temp[0])
            out = self.RNN[0].pool(out)
            data = self.RNN[0].pool(data)
            temp = self.RNN[0].pool(temp)

            # 1
            short = out
            short_data = data
            short_temp = temp
            out, initial[1], _, data, initial_single_temp[1], temp = self.RNN[1](out, initial[1], data, temp,
                                                                                 initial_single_temp[1])
            out, initial[2], _, data, initial_single_temp[2], temp = self.RNN[2](out, initial[2], data, temp,
                                                                                 initial_single_temp[2])
            out = out + short
            data = data + short_data
            temp = temp + short_temp

            # 3
            short = out
            short_data = data
            short_temp = temp
            out, initial[3], _, data, initial_single_temp[3], temp = self.RNN[3](out, initial[3], data, temp,
                                                                                 initial_single_temp[3])
            out, initial[4], data_map, data, initial_single_temp[4], temp = self.RNN[4](out, initial[4], data, temp,
                                                                                        initial_single_temp[4])
            out = out + short
            data = data + short_data
            temp = temp + short_temp
            temp_map_4[:, frame] = self.ofm[0](temp)

            out = self.dropout_RNN(out)
            data = self.dropout_RNN(data)
            temp = self.dropout_RNN(temp)

            # 5
            short = out
            short_data = data
            short_temp = temp
            out, initial[5], _, data, initial_single_temp[5], temp = self.RNN[5](out, initial[5], data, temp,
                                                                                 initial_single_temp[5])
            out, initial[6], _, data, initial_single_temp[6], temp = self.RNN[6](out, initial[6], data, temp,
                                                                                 initial_single_temp[6])
            out = out + self.ShortCut[0](short)
            data = data + self.ShortCut[0](short_data)
            temp = temp + self.ShortCut[0](short_temp)

            # 7
            short = out
            short_data = data
            short_temp = temp
            out, initial[7], _, data, initial_single_temp[7], temp = self.RNN[7](out, initial[7], data, temp,
                                                                                 initial_single_temp[7])
            out, initial[8], _, data, initial_single_temp[8], temp = self.RNN[8](out, initial[8], data, temp,
                                                                                 initial_single_temp[8])
            out = out + short
            data = data + short_data
            temp = temp + short_temp
            temp_map_8[:, frame] = self.ofm[1](temp)

            # 9
            short = out
            short_data = data
            short_temp = temp
            out, initial[9], _, data, initial_single_temp[9], temp = self.RNN[9](out, initial[9], data, temp,
                                                                                 initial_single_temp[9])
            out, initial[10], _, data, initial_single_temp[10], temp = self.RNN[10](out, initial[10], data, temp,
                                                                                    initial_single_temp[10])
            out = out + self.ShortCut[1](short)
            data = data + self.ShortCut[1](short_data)
            temp = temp + self.ShortCut[1](short_temp)

            out = self.dropout_RNN(out)
            data = self.dropout_RNN(data)
            temp = self.dropout_RNN(temp)

            # 11
            short = out
            short_data = data
            short_temp = temp
            out, initial[11], _, data, initial_single_temp[11], temp = self.RNN[11](out, initial[11], data, temp,
                                                                                    initial_single_temp[11])
            out, initial[12], _, data, initial_single_temp[12], temp = self.RNN[12](out, initial[12], data, temp,
                                                                                    initial_single_temp[12])
            out = out + short
            data = data + short_data
            temp = temp + short_temp
            temp_map_12[:, frame] = self.ofm[2](temp)

            # 13
            short = out
            short_data = data
            short_temp = temp
            out, initial[13], _, data, initial_single_temp[13], temp = self.RNN[13](out, initial[13], data, temp,
                                                                                    initial_single_temp[13])
            out, initial[14], _, data, initial_single_temp[14], temp = self.RNN[14](out, initial[14], data, temp,
                                                                                    initial_single_temp[14])
            out = out + self.ShortCut[2](short)
            data = data + self.ShortCut[2](short_data)
            temp = temp + self.ShortCut[2](short_temp)

            # 15
            short = out
            short_data = data
            short_temp = temp
            out, initial[15], _, data, initial_single_temp[15], temp = self.RNN[15](out, initial[15], data, temp,
                                                                                    initial_single_temp[15])
            out, initial[16], _, data, initial_single_temp[16], temp = self.RNN[16](out, initial[16], data, temp,
                                                                                    initial_single_temp[16])
            out = out + short
            data = data + short_data
            temp = temp + short_temp
            temp_map_16[:, frame] = self.ofm[3](temp)

            out = self.dropout_RNN(out)
            data = self.dropout_RNN(data)

            # out = out.detach()
            if not self.training:
                out_action[:, frame] = self.sftmx(2*self.classifier(out.contiguous().view(x.shape[0], -1)))
                data_action[:, frame] = self.sftmx(2*self.classifier(data.contiguous().view(x.shape[0], -1)))

            else:
                out_action[:, frame] = self.sftmx(self.classifier(out.contiguous().view(x.shape[0], -1)))
                data_action[:, frame] = self.sftmx(self.classifier(data.contiguous().view(x.shape[0], -1)))

        if self.test_scheme == 1:
            out = torch.mean(out_action, 1)
            data = torch.mean(data_action, 1)
        elif self.test_scheme == 2:
            if self.training:
                out = torch.mean(out_action[:, int(out_action.shape[1]/2):out_action.shape[1]],1)
                data = torch.mean(data_action[:, int(data_action.shape[1] / 2):data_action.shape[1]], 1)
            else:
                out = torch.mean(out_action, 1)
                data = torch.mean(data_action, 1)
        else:
            print("Wrong test_scheme")

        for i in range(len(initial)):
            initial[i] = initial[i].detach()
        for i in range(len(initial_single_temp)):
            initial_single_temp[i] = initial_single_temp[i].detach()
        temp_maps = [temp_map_4, temp_map_8, temp_map_12, temp_map_16]
        return out, out_action, data, data_action, temp_maps, initial, initial_single_temp


class OFM(nn.Module):
    def __init__(self, in_channel, mid_channel):
        super(OFM, self).__init__()
        self._make_block(in_channel, mid_channel)

    def _make_block(self, in_channel, mid_channel):
        self.a = nn.Conv2d(in_channel, mid_channel,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.a_bn = nn.BatchNorm2d(mid_channel)
        self.a_relu = nn.ReLU(inplace=True)

        self.b = nn.Conv2d(mid_channel, mid_channel,
                           kernel_size=3,
                           stride=1,
                           padding=1)
        self.b_bn = nn.BatchNorm2d(mid_channel)
        self.b_relu = nn.ReLU(inplace=True)

        self.c = nn.Conv2d(mid_channel, 2,
                           kernel_size=1,
                           stride=1,
                           padding=0)
        self.c_bn = nn.BatchNorm2d(2)
        self.c_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.a_relu(self.a_bn(self.a(x)))
        x = self.b_relu(self.b_bn(self.b(x)))
        x = self.c_relu(self.c_bn(self.c(x)))
        return x


def actionModel(num_action, batch_norm=False, dropout=[0, 0], test_scheme=1, q=0.5, image_size=112, syn_bn=False):

    model = SCS(batch_norm, num_action, dropout=dropout, test_scheme=test_scheme, q=q, img_size=image_size, syn_bn=syn_bn)

    return model


