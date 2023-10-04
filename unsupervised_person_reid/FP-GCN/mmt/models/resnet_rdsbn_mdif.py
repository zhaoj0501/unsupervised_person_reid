import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import init, Module, Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple
from collections import OrderedDict
import operator
from itertools import islice
from mmt.models.dsbn import DomainSpecificBatchNorm1d as BatchNorm1d
from mmt.models.rdsbn import DomainSpecificRcBatchNorm2d as RDSBN2d


_pair = _ntuple(2)

__all__ = ['resnet50_rdsbn_mdif']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        setattr(self.weight, 'gcn_weight', True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            setattr(self.bias, 'gcn_weight', True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_2ADJ(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN_2ADJ, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj_1, adj_2):
        x = self.gc1(x, adj_1)
        x = self.relu(x)
        x = self.gc2(x, adj_2)
        return x


class ADJ_FirstLayer(nn.Module):
    def __init__(self):
        super(ADJ_FirstLayer, self).__init__()

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def forward(self, x):
        N, C = x.size()

        adj = torch.eye(N + 4).cuda()
        adj[N:, N:] = 1.0

        adj = self.gen_adj(adj)
        return adj


class ADJ_AttCenter(nn.Module):
    def __init__(self):
        super(ADJ_AttCenter, self).__init__()
        self.attention = nn.Linear(2048, 1)

        self.register_buffer('running_mean', torch.zeros(4, 2048))
        self.momentum = 0.9

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)
        return adj

    def forward(self, x):
        N, C = x.size()

        if self.training:
            stride = N // 4
            weight = self.attention(x)      # (N, 1)
            weight0, weight1, weight2, weight3 = weight[:stride], weight[stride: 2 * stride], \
                                                 weight[2 * stride: 3 * stride], weight[3 * stride:]
            weight0 = weight0 / weight0.sum(dim=0, keepdim=False)
            weight1 = weight1 / weight1.sum(dim=0, keepdim=False)
            weight2 = weight2 / weight2.sum(dim=0, keepdim=False)
            weight3 = weight3 / weight3.sum(dim=0, keepdim=False)

            center0 = torch.sum(x[:stride] * weight0, dim=0, keepdim=True)
            center1 = torch.sum(x[stride: 2 * stride] * weight1, dim=0, keepdim=True)
            center2 = torch.sum(x[2 * stride: 3 * stride] * weight2, dim=0, keepdim=True)
            center3 = torch.sum(x[3 * stride:] * weight3, dim=0, keepdim=True)

            x_new = torch.cat([x, center0, center1, center2, center3], dim=0)

            # added for test
            self.running_mean.mul_(self.momentum)
            self.running_mean[0].add_((1 - self.momentum) * center0[0].data)
            self.running_mean[1].add_((1 - self.momentum) * center1[0].data)
            self.running_mean[2].add_((1 - self.momentum) * center2[0].data)
            self.running_mean[3].add_((1 - self.momentum) * center3[0].data)
            # added for test

            adj = torch.eye(N + 4).cuda()
            adj[N:, N:] = 1.

            # origin
            adj[:stride, N] = 1.
            adj[N, :stride] = 1.

            adj[stride: 2 * stride, N+1] = 1.
            adj[N+1, stride: 2 * stride] = 1.

            adj[2 * stride: 3 * stride, N + 2] = 1.
            adj[N + 2, 2 * stride: 3 * stride] = 1.

            adj[3 * stride:, N + 3] = 1.
            adj[N + 3, 3 * stride:] = 1.

        else:
            x_new = torch.cat([x, torch.autograd.Variable(self.running_mean)], dim=0)
            adj = torch.eye(N + 4).cuda()
            adj[N:, N:] = 1.

            adj[:N, N] = 1.
            adj[N, :N] = 1.

        adj = self.gen_adj(adj)
        return x_new, adj


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias,
            padding_mode='zeros'
        )

    def forward(self, input, domain_label):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups), domain_label


class TwoInputSequential(nn.Module):
    r"""A sequential container forward with two inputs.
    """

    def __init__(self, *args):
        super(TwoInputSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx):
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TwoInputSequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx, module):
        key = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx):
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)

    def __len__(self):
        return len(self._modules)

    def __dir__(self):
        keys = super(TwoInputSequential, self).__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2


def resnet50_rdsbn_mdif(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RDSBNResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        updated_state_dict = _update_initial_weights_dsbn(model_zoo.load_url(model_urls['resnet50']),
                                                          num_classes=model.num_classes,
                                                          num_domains=model.num_domains)
        model.load_state_dict(updated_state_dict, strict=False)

    return model


def _update_initial_weights_dsbn(state_dict, num_classes=1000, num_domains=2, dsbn_type='all'):
    new_state_dict = state_dict.copy()

    for key, val in state_dict.items():
        update_dict = False
        if ((('bn' in key or 'downsample.1' in key) and dsbn_type == 'all') or
                (('bn1' in key) and dsbn_type == 'partial-bn1')):
            update_dict = True

        if (update_dict):
            if 'weight' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-6] + 'bns.{}.weight'.format(d)] = val.data.clone()

            elif 'bias' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-4] + 'bns.{}.bias'.format(d)] = val.data.clone()

            if 'running_mean' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-12] + 'bns.{}.running_mean'.format(d)] = val.data.clone()

            if 'running_var' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-11] + 'bns.{}.running_var'.format(d)] = val.data.clone()

            if 'num_batches_tracked' in key:
                for d in range(num_domains):
                    new_state_dict[
                        key[0:-len('num_batches_tracked')] + 'bns.{}.num_batches_tracked'.format(d)] = val.data.clone()

    if num_classes != 1000 or len([key for key in new_state_dict.keys() if 'fc' in key]) > 1:
        key_list = list(new_state_dict.keys())
        for key in key_list:
            if 'fc' in key:
                print('pretrained {} are not used as initial params.'.format(key))
                del new_state_dict[key]

    return new_state_dict


def update_source_weight(state_dict):
    new_state_dict = state_dict.copy()
    for key, val in state_dict.items():
        if 'bns.3' in key:
            new_key = key.replace('bns.3', 'bns.0')
            print('Copying bn weights from {} to {}'.format(key, new_key))
            new_state_dict[new_key] = val.data.clone()
    return new_state_dict


class RDSBNResNet(nn.Module):
    def __init__(self, block, layers, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, num_domains=2, within_single_batch=True):
        super(RDSBNResNet, self).__init__()
        # init setting
        self.inplanes = 64
        self.num_features = num_features
        self.num_classes = num_classes
        self.cut_at_pooling = cut_at_pooling
        self.norm = norm
        self.dropout = dropout
        self.has_embedding = num_features > 0
        self.num_domains = num_domains
        self.within_single_batch = within_single_batch

        # build base network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)   # share
        self.bn1 = RDSBN2d(64, self.num_domains, within_single_batch=self.within_single_batch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], num_domains=self.num_domains,
                                       within_single_batch=self.within_single_batch)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, num_domains=self.num_domains,
                                       within_single_batch=self.within_single_batch)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, num_domains=self.num_domains,
                                       within_single_batch=self.within_single_batch)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, num_domains=self.num_domains,
                                       within_single_batch=self.within_single_batch)   # last stride 1
        self.gap = nn.AdaptiveAvgPool2d(1)

        if not self.cut_at_pooling:
            out_planes = 512 * block.expansion

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = BatchNorm1d(self.num_features, self.num_domains)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = BatchNorm1d(self.num_features, self.num_domains)
            self.feat_bn.reset_parameters()
            self.feat_bn.bias_requires_grad(False)

            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                init.normal_(self.classifier.weight, std=0.001)

        self.gcn = GCN_2ADJ(self.num_features, self.num_features, self.num_features)
        self.adj_1 = ADJ_FirstLayer()
        self.adj_2 = ADJ_AttCenter()
        self.gcn_bn = BatchNorm1d(self.num_features, self.num_domains)
        self.gcn_classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
        self.gcn_bn.reset_parameters()
        self.gcn_bn.bias_requires_grad(False)

        self.reset_params()

    def forward(self, x, domain_label, feature_withbn=False, epochs=-1):
        domain_label_temp = torch.cat([domain_label,
                                       epochs * torch.ones(1, dtype=torch.long).cuda()], dim=0)

        x = self.conv1(x)
        x, _ = self.bn1(x, domain_label_temp)
        # x = self.relu(x)
        x = self.maxpool(x)

        x, _ = self.layer1(x, domain_label_temp)
        x, _ = self.layer2(x, domain_label_temp)
        x, _ = self.layer3(x, domain_label_temp)
        x, _ = self.layer4(x, domain_label_temp)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # MDIF
        adj_1 = self.adj_1(x)
        x_new, adj_2 = self.adj_2(x)
        adj_1 = adj_1.detach()
        adj_2 = adj_2.detach()

        gcn_feat = self.gcn(x_new, adj_1, adj_2)
        x_gcn = x + gcn_feat[:x.size()[0]]
        x_gcn_bn, _ = self.gcn_bn(x_gcn, domain_label)
        gcn_prob = self.gcn_classifier(x_gcn_bn)

        if self.training is False:
            norm_x = F.normalize(x_gcn_bn)
            return norm_x

        if self.has_embedding:
            bn_x, _ = self.feat_bn(self.feat(x), domain_label)
        else:
            bn_x, _ = self.feat_bn(x, domain_label)

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(bn_x)
        else:
            return x, bn_x

        if feature_withbn:
            return bn_x, prob
        return x, prob, gcn_prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, RDSBN2d):
                m.reset_parameters()

    def _make_layer(self, block, planes, blocks, stride=1, num_domains=2, within_single_batch=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = TwoInputSequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                RDSBN2d(planes * block.expansion, num_domains, within_single_batch=within_single_batch),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            num_domains=num_domains, within_single_batch=within_single_batch))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes,
                      num_domains=num_domains, within_single_batch=within_single_batch))

        return TwoInputSequential(*layers)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_domains=2, within_single_batch=False):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = RDSBN2d(planes, num_domains, within_single_batch=within_single_batch)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = RDSBN2d(planes, num_domains, within_single_batch=within_single_batch)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = RDSBN2d(planes * 4, num_domains, within_single_batch=within_single_batch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, domain_label):
        residual = x

        out = self.conv1(x)
        out, _ = self.bn1(out, domain_label)
        out = self.relu(out)

        out = self.conv2(out)
        out, _ = self.bn2(out, domain_label)
        out = self.relu(out)

        out = self.conv3(out)
        out, _ = self.bn3(out, domain_label)

        if self.downsample is not None:
            residual, _ = self.downsample(x, domain_label)

        out += residual
        out = self.relu(out)

        return out, domain_label
