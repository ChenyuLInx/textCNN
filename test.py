#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created on Wedns Sep 6 23:03 2017

@auther Chenyu Li
"""
import sys
import torch
from torch import autograd
import math
import argparse
from data_utils import construct_vocab, convert2tensor
from model import EncoderCNN, DecoderCNN


def test():
    encoder = EncoderCNN(60,200,5,2,[300,100,200],12)

    decoder = DecoderCNN(200,5,2,[300,100,200],12)
    data = torch.randperm(57).view(1,57)
    input = autograd.Variable(torch.cat((data,data,data)))
    hidden,embed = encoder(input)
    final = decoder(hidden)
    import code
    code.interact(local=locals())
    # TODO Load training data

    # TODO train data
    # TODO validate result
    # TODO save model


if __name__ == "__main__":
    test()
