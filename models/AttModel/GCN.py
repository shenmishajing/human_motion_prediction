#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
adapted from : https://github.com/wei-mao-2019/HisRepItself/blob/master/model/GCN.py
under MIT license.
"""

from __future__ import absolute_import, print_function

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = input.matmul( self.weight)  # w: 40, 256 -> 256,256 -> 256,40
        output = self.att.matmul( support)  # att: 39*39
        if self.bias is not None:
            output= output + self.bias
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.dropout = nn.Dropout(p_dropout)
        self.act = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.dropout(self.act(self.bn1(y.reshape(b, -1)).reshape(b, n, f)))

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.dropout(self.act(self.bn2(y.reshape(b, -1)).reshape(b, n, f)))

        return y + x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(nn.Module):
    def __init__(
        self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48
    ):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = nn.ModuleList()
        for _ in range(num_stage):
            self.gcbs.append(
                GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n)
            )

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.dropout = nn.Dropout(p_dropout)
        self.act = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.dropout(self.act(self.bn(y.reshape(b, -1)).reshape(b, n, f)))

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        return self.gc7(y) + x
