# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
import math
import numpy as np

from mmpose.core import WeightNormClipHook
from ..builder import BACKBONES
from .base_backbone import BaseBackbone

from .utils.graph_skeleton import Graph_Skeleton
from .utils.graph_utils import adj_mx_from_skeleton


class SemCHGraphConv(nn.Module):
    """
    Semantic channel-wise graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=False):
        super(SemCHGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

      
        self.adj = adj.unsqueeze(0).repeat(out_features, 1, 1)
        self.m = (self.adj > 0)
        self.e = nn.Parameter(torch.zeros(out_features, len(self.m[0].nonzero()), dtype=torch.float))
        nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(1))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        # input: (B, T, K, C)
        h0 = torch.matmul(input, self.W[0]).unsqueeze(2).transpose(2, 4)  # B * T * C * K * 1
        h1 = torch.matmul(input, self.W[1]).unsqueeze(2).transpose(2, 4)  # B * T * C * K * 1

        adj = -9e15 * torch.ones_like(self.adj).to(input.device)  # C * J * J = 128*9*9
        adj[self.m] = self.e.view(-1)
        adj = F.softmax(adj, dim=2)

        E = torch.eye(adj.size(1), dtype=torch.float).to(input.device)
        E = E.unsqueeze(0).repeat(self.out_features, 1, 1)  # C * J * J

        output = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
        output = output.transpose(2, 4).squeeze(2)

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class LocalGraph(nn.Module):
    def __init__(self, adj, input_dim, output_dim, dropout=None):
        super(LocalGraph, self).__init__()

        num_joints = adj.shape[0]
        if num_joints == 9:
            store_2 = [7, 8] 
            joints_left = [1, 2, 3, 7]
            joints_right = [4, 5, 6, 8]
        else:
            raise KeyError("The dimension of adj matrix is wrong!")


        adj_sym = torch.zeros_like(adj)
        for i in range(num_joints):
            for j in range(num_joints):
                if i == j:
                    adj_sym[i][j] = 1
                if i in joints_left:
                    index = joints_left.index(i)
                    adj_sym[i][joints_right[index]] = 1.0
                if i in joints_right:
                    index = joints_right.index(i)
                    adj_sym[i][joints_left[index]] = 1.0

        adj_1 = adj.matrix_power(1)
        for i in np.arange(num_joints):
            if i in store_2:
                adj_1[i] = 0

        adj_2 = adj.matrix_power(2)
        # store_2 = [3, 6, 10, 13, 16]
        for i in np.arange(num_joints):
            if i not in store_2:
                adj_2[i] = 0

        adj_con = adj_1 + adj_2

        self.gcn_sym = SemCHGraphConv(input_dim, output_dim, adj_sym) 
        self.bn_1 = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.gcn_con = SemCHGraphConv(input_dim, output_dim, adj_con) 
        self.bn_2 = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.relu = nn.ReLU()

        self.cat_conv = nn.Conv2d(2 * output_dim, output_dim, 1, bias=False)
        self.cat_bn = nn.BatchNorm2d(output_dim, momentum=0.1)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, input):
        # x: (B, T, K, C)
        # x = self.gcn_sym(input)
        y = self.gcn_con(input)

        y = y.permute(0, 3, 1, 2)
        y = self.relu(self.bn_2(y))

        # x: (B, T, K, C) --> (B, C, T, K)
        # x = x.permute(0, 3, 1, 2)

        # x = self.relu(self.bn_1(x))

        # output = torch.cat((x, y), dim=1)
        # output = self.cat_bn(self.cat_conv(output)) 

        if self.dropout is not None:
            output = self.dropout(y)
        else:
            output = y
        output = output.permute(0, 2, 3, 1)

        return output



class GlobalGraphConv(nn.Module):
    """"
    Global graph attention layer
    """

    def __init__(self, adj, in_channels, inter_channels=None):
        super(GlobalGraphConv, self).__init__()

        self.adj = adj
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.2)

        if self.inter_channels == self.in_channels // 2:
            self.g_channels = self.in_channels
        else:
            self.g_channels = self.inter_channels

        assert self.inter_channels > 0

        self.g = nn.Conv1d(in_channels=self.in_channels, out_channels=self.g_channels,
                           kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv1d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        adj_shape = self.adj.shape
        self.C_k = nn.Parameter(torch.zeros(adj_shape, dtype=torch.float)) 

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
        )

        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)

    def forward(self, x):
        batch_size = x.size(0)  # x: (B*T, C, N) 

        # g_x: (B*T, N, C/k)
        g_x = self.g(x).view(batch_size, self.g_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # (B*T, C/k, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
        # (B*T, C/k, 1, N)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        # h: N, w: N
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.expand(-1, -1, -1, w)  # (B*T, C/k, N, N) 
        phi_x = phi_x.expand(-1, -1, h, -1)

        # concat_feature: (B*T, C/k, N, N)
        concat_feature = torch.cat([theta_x, phi_x], dim=1) 
        f = self.concat_project(concat_feature)  # (B*T, 1, N, N)
        b, _, h, w = f.size()
        f = f.view(b, h, w)
        attention = self.leakyrelu(f)  # (B*T, N, N)  attention:B_k  

        attention = torch.add(self.softmax(attention), self.C_k) 
        # y: (B*T, C/k, N)
        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.g_channels, *x.size()[2:])

        return y


class GlobalGraph(nn.Module):
    def __init__(self, adj, in_channels, output_channels, dropout=None):
        super(GlobalGraph, self).__init__()

        self.attentions = GlobalGraphConv(adj, in_channels, output_channels // 2)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # x: (B, T, K, C) --> (B*T, K, C)
        x_size = x.shape
        x = x.contiguous()
        x = x.view(-1, *x_size[2:])
        # x: (B*T, C, K)
        x = x.permute(0, 2, 1)

        x = self.attentions(x)

        # x: (B*T, C, K) --> (B*T, K, C)
        x = x.permute(0, 2, 1).contiguous()

        # x = torch.matmul(x, self.W)
        # x: (B*T, K, C) --> (B, T, K, C)
        x = x.view(*x_size)

        # x: (B, T, K, C) --> (B, C, T, K)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.bn(x))

        if self.dropout is not None:
            x = self.dropout(x)

        # x: (B, C, T, K) --> (B, T, K, C)
        x = x.permute(0, 2, 3, 1)

        return x



class MultiGlobalGraph(nn.Module):
    def __init__(self, adj, in_channels, inter_channels, dropout=None):
        super(MultiGlobalGraph, self).__init__()

        self.num_non_local = in_channels // inter_channels

        attentions = [GlobalGraphConv(adj, in_channels, inter_channels) for _ in range(self.num_non_local)]
        self.attentions = nn.ModuleList(attentions)

        self.cat_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.cat_bn = nn.BatchNorm2d(in_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        # x: (B, T, K, C) --> (B*T, K, C)
        residual = x  # lsj-Non Local
        x_size = x.shape
        x = x.contiguous()
        x = x.view(-1, *x_size[2:])
        # x: (B*T, C, K)
        x = x.permute(0, 2, 1)

        x = torch.cat([self.attentions[i](x) for i in range(len(self.attentions))], dim=1)

        # x: (B*T, C, K) --> (B*T, K, C)
        x = x.permute(0, 2, 1).contiguous()

        # x = torch.matmul(x, self.W)
        # x: (B*T, K, C) --> (B, T, K, C)
        x = x.view(*x_size)

        x = x + residual  # lsj-Non Local

        # x: (B, T, K, C) --> (B, C, T, K)
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.cat_bn(self.cat_conv(x)))

        if self.dropout is not None:
            x = self.dropout(x)

        # x: (B, C, T, K) --> (B, T, K, C)
        x = x.permute(0, 2, 3, 1)

        return x



class GraphAttentionBlock(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=0.25):
        super(GraphAttentionBlock, self).__init__()

        hid_dim = output_dim
        self.relu = nn.ReLU(inplace=True)
        self.adj = adj.matrix_power(2)

        self.local_graph_layer = LocalGraph(adj, input_dim, hid_dim, p_dropout)

        self.global_graph_layer = MultiGlobalGraph(adj, input_dim, input_dim // 4, dropout=p_dropout)
        # self.global_graph_layer = GlobalGraph(adj, input_dim, output_dim)

        # self.cat_conv = nn.Conv2d(3 * output_dim, 2 * output_dim, 1, bias=False)
        # self.cat_bn = nn.BatchNorm2d(2 * output_dim, momentum=0.1)
        self.cat_conv = nn.Conv2d(2 * output_dim, output_dim, 1, bias=False)
        self.cat_bn = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.conv1d = nn.Conv2d(output_dim, output_dim, 1, bias=False)

    def forward(self, x):
        residual = x

        # x: (B, C, T, K) --> (B, T, K, C)    
        x = x.permute(0, 2, 3, 1)
        x_ = self.local_graph_layer(x)
        y_ = self.global_graph_layer(x)
        x = torch.cat((x_, y_), dim=-1) 

        # x: (B, T, K, C) --> (B, C, T, K) 
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.cat_bn(self.cat_conv(x))) + residual   # channel callback lsj
        return x



class GraphAttentionBlock2(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=0.25):
        super(GraphAttentionBlock2, self).__init__()

        hid_dim = output_dim
        self.relu = nn.ReLU(inplace=True)
        self.adj = adj.matrix_power(2)

        self.local_graph_layer = LocalGraph(adj, input_dim, hid_dim, p_dropout)
        self.global_graph_layer = MultiGlobalGraph(adj, input_dim, input_dim // 4, dropout=p_dropout)
        # self.global_graph_layer = GlobalGraph(adj, input_dim, output_dim)

        # self.cat_conv = nn.Conv2d(3 * output_dim, 2 * output_dim, 1, bias=False)
        # self.cat_bn = nn.BatchNorm2d(2 * output_dim, momentum=0.1)
        self.cat_conv = nn.Conv2d(2 * output_dim, output_dim, 1, bias=False) 
        self.cat_bn = nn.BatchNorm2d(output_dim, momentum=0.1)
        self.conv1d = nn.Conv2d(output_dim, output_dim, 1, bias=False)

    def forward(self, x):
        residual = x

        # x: (B, C, T, K) --> (B, T, K, C)     
        x = x.permute(0, 2, 3, 1)
        x_ = self.local_graph_layer(x)
        y_ = self.global_graph_layer(x)
        x = torch.cat((x_, y_), dim=-1) 

        # x: (B, T, K, C) --> (B, C, T, K)   
        x = x.permute(0, 3, 1, 2)
        x = self.relu(self.cat_bn(self.cat_conv(x) + residual))   # channel callback wyh
        return x



class BasicTemporalBlock(nn.Module):
    """Basic block for VideoPose3D.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        mid_channels (int): The output channels of conv1. Default: 1024.
        kernel_size (int): Size of the convolving kernel. Default: 3.
        dilation (int): Spacing between kernel elements. Default: 3.
        dropout (float): Dropout rate. Default: 0.25.
        causal (bool): Use causal convolutions instead of symmetric
            convolutions (for real-time applications). Default: False.
        residual (bool): Use residual connection. Default: True.
        use_stride_conv (bool): Use optimized TCN that designed
            specifically for single-frame batching, i.e. where batches have
            input length = receptive field, and output length = 1. This
            implementation replaces dilated convolutions with strided
            convolutions to avoid generating unused intermediate results.
            Default: False.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: dict(type='Conv1d').
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN1d').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=1024,
                 kernel_size=3,
                 dilation=3,
                 dropout=0.25,
                 causal=False,
                 residual=True,
                 use_stride_conv=False,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d')):
        # Protect mutable default arguments
        conv_cfg = copy.deepcopy(conv_cfg)
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.causal = causal
        self.residual = residual
        self.use_stride_conv = use_stride_conv

        self.pad = (kernel_size - 1) * dilation // 2
        if use_stride_conv:
            self.stride = kernel_size
            self.causal_shift = kernel_size // 2 if causal else 0
            self.dilation = 1
        else:
            self.stride = 1
            self.causal_shift = kernel_size // 2 * dilation if causal else 0

        self.conv1 = nn.Sequential(
            ConvModule(
                in_channels,
                mid_channels,
                kernel_size=(kernel_size, 1),
                stride=(self.stride, 1),
                dilation=(self.dilation, 1),
                bias='auto',
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))
        self.conv2 = nn.Sequential(
            ConvModule(
                mid_channels,
                out_channels,
                kernel_size=(1, 1),
                bias='auto',
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg))

        if residual and in_channels != out_channels:
            self.short_cut = build_conv_layer(conv_cfg, in_channels,
                                              out_channels, 1)
        else:
            self.short_cut = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """Forward function."""
        if self.use_stride_conv:
            assert self.causal_shift + self.kernel_size // 2 < x.shape[2]
        else:
            assert 0 <= self.pad + self.causal_shift < x.shape[2] - \
                   self.pad + self.causal_shift <= x.shape[2]

        out = self.conv1(x)
        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.dropout is not None:
            out = self.dropout(out)

        if self.residual:
            if self.use_stride_conv:
                res = x[:, :, self.causal_shift +
                              self.kernel_size // 2::self.kernel_size]
            else:
                res = x[:, :,
                      (self.pad + self.causal_shift):(x.shape[2] - self.pad +
                                                      self.causal_shift)]

            if self.short_cut is not None:
                res = self.short_cut(res)
            out = out + res

        return out


@BACKBONES.register_module()
class TSGCN_arm9(BaseBackbone): 
    """TSGCN backbone.
    Temporal Attention Graph Convolutional Networks.

    Args:
        in_channels (int): Number of input channels, which equals to
            num_keypoints * num_features.
        stem_channels (int): Number of feature channels. Default: 1024
        num_blocks (int): NUmber of basic temporal convolutional blocks.
            Default: 2.
        kernel_sizes (Sequence[int]): Sizes of the convolving kernel of
            each basic block. Default: ``(3, 3, 3)``.
        dropout (float): Dropout rate. Default: 0.25.
        causal (bool): Use causal convolutions instead of symmetric
            convolutions (for real-time applications).
            Default: False.
        residual (bool): Use residual connection. Default: True.
        use_stride_conv (bool): Use TCN backbone optimized for single-frame batching,
            i.e. where batches have input length = receptive field, and output length = 1.
            This implementation replaces dilated convolutions with strided convolutions avoid
            generating unused intermediate results. The weights are interchangeable with
            the reference implementation. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: dict(type='Conv1d').
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN1d').
        max_norm (float|None): if not None, the weight of convolution layers
            will be clipped to have a maximum norm of max_norm.

    Example:
        >>> from mmpose.models import TCN
        >>> import torch
        >>> self = TCN(in_channels=34)
        >>> self.eval()
        >>> inputs = torch.rand(1, 34, 243)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 1024, 235)
        (1, 1024, 217)
    """

    def __init__(self,
                 in_channels=2,
                 stem_channels=1024,  # wyh 1024
                 num_blocks=3,
                 kernel_sizes=(3, 3, 3, 3),
                 dropout=0.25,
                 causal=False,
                 residual=True,
                 num_joints=9,
                 use_stride_conv=False,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'), 
                 max_norm=None):
        conv_cfg = copy.deepcopy(conv_cfg)
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__()
        self.in_channels = in_channels
        self.stem_channels = stem_channels
        self.num_blocks = num_blocks
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.causal = causal
        self.residual = residual
        self.use_stride_conv = use_stride_conv
        self.max_norm = max_norm
        self.num_joints = num_joints

        assert num_blocks == len(kernel_sizes) - 1
        for ks in kernel_sizes:
            assert ks % 2 == 1, 'Only odd filter widths are supported.'


        self.expand_conv = ConvModule(
            in_channels,   # 2
            stem_channels,  # default 1024
            kernel_size=(kernel_sizes[0], 1), 
            stride=(kernel_sizes[0], 1) if use_stride_conv else 1,
            bias='auto',
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        # ----------------- adj ---------------------------------------------
        arm9_skeleton = Graph_Skeleton(parents=[-1, 0, 1, 2, 0, 4, 5, 3, 6],  # 9
                                       joints_left=[1, 2, 3, 7], joints_right=[4, 5, 6, 8])
        self.adj = adj_mx_from_skeleton(arm9_skeleton) 
        self.graphConv_blocks = nn.ModuleList()
        self.graphConv_blocks.append(
            GraphAttentionBlock(adj=self.adj, input_dim=stem_channels,
                                output_dim=stem_channels, p_dropout=dropout)  # 输出channel= 2*output_dim
        )


        dilation = kernel_sizes[0]
        # -------------------- TCN  BLOCK ----------------------------
        self.tcn_blocks = nn.ModuleList()
        for i in range(1, num_blocks + 1):
            self.tcn_blocks.append(
                BasicTemporalBlock(    
                    in_channels=stem_channels,   # wyh  in_channels=stem_channels
                    out_channels=stem_channels,  # out_channels=stem_channels
                    mid_channels=stem_channels,  # mid_channels=stem_channels
                    kernel_size=kernel_sizes[i],
                    dilation=dilation,
                    dropout=dropout,
                    causal=causal,
                    residual=residual,
                    use_stride_conv=use_stride_conv,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
            dilation *= kernel_sizes[i]  # 3, 9, 27, 81, 243
        # -------------------- GA BLOCK ----------------------------
        for i in range(1, num_blocks):
            self.graphConv_blocks.append(
                GraphAttentionBlock(adj=self.adj, input_dim=stem_channels,
                                    output_dim=stem_channels, p_dropout=dropout)
            )
        self.graphConv_blocks.append(
            GraphAttentionBlock2(adj=self.adj, input_dim=stem_channels,
                                output_dim=stem_channels, p_dropout=dropout)
        )

        if self.max_norm is not None:
            # Apply weight norm clip to conv layers
            weight_clip = WeightNormClipHook(self.max_norm)
            for module in self.modules():
                if isinstance(module, nn.modules.conv._ConvNd):
                    weight_clip.register(module)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        """Forward function."""
        # (128,channel=18,size=81) -> (128,9,2,81)
        x = x.permute(0, 2, 1)     # (b, size, channel)
        x = x.reshape([x.shape[0], x.shape[1], self.num_joints, 2])  # (b, size=T, K=9, channel=2)
        x = x.permute(0, 3, 1, 2)  # (B, channel=2, size=T, K=9)
        x = self.expand_conv(x)
        x = self.graphConv_blocks[0](x)

        # if self.dropout is not None:
        #     x = self.dropout(x)

        outs = []
        for i in range(self.num_blocks):
            x = self.tcn_blocks[i](x)
            x = self.graphConv_blocks[i+1](x)
            outs.append(x)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        """Initialize the weights."""
        super().init_weights(pretrained)
        if pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.modules.conv._ConvNd):
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
