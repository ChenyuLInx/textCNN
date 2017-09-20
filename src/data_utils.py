#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 22:30:34 2017

@author: chenyu
"""

import torch
from torch.autograd import Variable
from nltk.tokenize import word_tokenize
import operator
import json


def construct_vocab(lines, vocab_size=50000):
    """Construct a vocabulary from tokenized lines."""
    vocab = {}
    docs = [word_tokenize(line) for line in lines]
    for doc in docs:
        for word in doc:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    word2id = {}
    id2word = {}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    id2word[0] = '<pad>'
    id2word[1] = '<pad>'

    sorted_word2id = sorted(
        vocab.items(),
        key=operator.itemgetter(1),
        reverse=True
    )

    sorted_words = [x[0] for x in sorted_word2id[:vocab_size]]

    for ind, word in enumerate(sorted_words):
        word2id[word] = ind + 2

    for ind, word in enumerate(sorted_words):
        id2word[ind + 2] = word

    return docs, word2id, id2word


def convert2tensor(batch, word2id, pad_to=60):
    """Prepare minibatch."""
    lens = [len(line) for line in batch]
    max_len = lens[-1]
    if pad_to > max_len:
        max_len = pad_to
    input_lines = [
        [word2id[w] if w in word2id else word2id['<unk>'] for w in doc] +
        [word2id['<pad>']] * (max_len - len(doc))
        for doc in batch
    ]

    tensor_batch = Variable(torch.LongTensor(input_lines))

    return tensor_batch, max_len

def load_txt(data_path):
    with open(data_path) as f:
        reviews = f.readlines()
    txt = []
    for line in reviews:
        data = json.loads(line)
        if len(data['text']) > 20:
            txt.append(data['text'])
    return txt

def test():
    test = ['this is a test', 'I have another test']
    docs, word2id, id2word = construct_vocab(test)
    print(word2id)
    print(id2word)
    test_batch, max_len = convert2tensor(docs, word2id)
    print(test_batch)
    print(max_len)


if __name__ == "__main__":
    test()
