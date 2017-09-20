#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created on Wedns Sep 6 23:03 2017

@auther Chenyu Li
"""
import torch
import math
import argparse
from data_utils import construct_vocab, convert2tensor
from model import EncoderCNN, DecoderCNN


def main():
    # TODO parse arguments
    parser = argparse.ArgumentParser()
    # DATA related
    parser.add_argument('--data_path', type=str,
                        help='Path of training data')
    parser.add_argument('--max_length', type=int, default=250,
                        help='Pad all data to this size')
    # Model Specs
    parser.add_argument('--embed_size', type=int, default=300,
                        help='Dimension of word embedding')
    parser.add_argument('--filter_size', type=int, default=5,
                        help='size of filter')
    parser.add_argument('--stride', type=int, default=2,
                        help='stride size for each layer')
    parser.add_argument('--filter_nums', type=str, default='300,600',
                        help='filer number for each convolution layer')
    parser.add_argument('--hidden_size', type=int, default=500,
                        help='size of hidden state in the middle')
    # Optimization Specs
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Minibatch during training')
    parser.add_argument('--optim', type=str, default='SGD',
                        help='Optimization method')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='Learning rate of model')
    parser.add_argument('--decay', type=float, default=0.5,
                        help='Decay rate of learning rate')
    parser.add_argument('--start_decay', type=int, default=5,
                        help='Start Epoch of decay learning rate')
    args = parser.parse_args()
    args.filter_nums = [int(i) for i in args.filter_nums.split(',')]
    args.filter_nums.append(args.hidden_size)
    final_filter_size = args.max_length
    for num in args.filter_nums[:-1]:
        final_filter_size = math.floor(
                    (final_filter_size - args.filter_size)/args.stride + 1
                    )
    # TODO build model
    encoder = EncoderCNN(args.embed_size, args.filter_size,
                         args.stride, args.filter_nums, final_filter_size)
    decoder = DecoderCNN(args.embed_size, args.filter_size,
                         args.stride, args.filter_nums, final_filter_size)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()
    creterion = 
    # TODO Load training data

    with open(args.data_path, 'r') as f:
        data = f.readlines()
    # TODO train data
    # TODO validate result
    # TODO save model


if __name__ == "__main__":
    main()
