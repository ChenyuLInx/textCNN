import torch
import torch.nn as nn
from torch.nn.init import xavier_normal

class EncoderCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, filter_size, stride,
                 filter_nums, final_filter_size):
        super(EncoderCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.layers = []
        self.layers.append(
                nn.Sequential(
                    nn.Conv1d(embed_size, filter_nums[0], 
                              filter_size, stride=stride),
                    nn.BatchNorm1d(filter_nums[0]),
                    nn.ReLU()
                    )
                )
        for i, filter_num in enumerate(filter_nums[1:-1]):
            self.layers.append(
                    nn.Sequential(
                        nn.Conv1d(filter_nums[i], filter_num, 
                                  filter_size, stride=stride),
                        nn.BatchNorm1d(filter_num),
                        nn.ReLU()
                        )
                    )

        self.layers.append(
                nn.Sequential(
                    nn.Conv1d(filter_nums[-2], filter_nums[-1],
                              final_filter_size),
                    nn.BatchNorm1d(filter_nums[-1])
                    )
                )

    def init_weights(self):
        """Initialize the weight of network"""
        def xavier_init(m):
            if type(m) == nn.Conv1d:
                xavier_normal(m.weight)
        for layer in self.layers:
            layer.apply(xavier_init)

    def forward(self, x):
        self.embed.weight.data = nn.functional.normalize(self.embed.weight.data)
        x_embed = self.embed(x)
        x_embed = x_embed.transpose(1,2)
        out = self.layers[0](x_embed)
        print(out.size())
        for i in range(len(self.layers) - 1):
            out = self.layers[i + 1](out)
            print(out.size())
        return out, x_embed

class DecoderCNN(nn.Module):
    def __init__(self, embed_size, filter_size, stride,
                 filter_nums, final_filter_size):
        super(DecoderCNN, self).__init__()
        self.layers = []
        self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(filter_nums[-1], filter_nums[-2],
                              final_filter_size),
                    nn.BatchNorm1d(filter_nums[-2]),
                    nn.ReLU()
                    )
                )
        for i, filter_num in enumerate(filter_nums[1:-1]):
            self.layers.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(filter_nums[-(i+2)], filter_nums[-(i+3)],
                                  filter_size, stride=stride),
                        nn.BatchNorm1d(filter_nums[-(i+3)]),
                        nn.ReLU()
                        )
                    )

        self.layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(filter_nums[0], embed_size,
                              filter_size, stride=stride),
                    nn.BatchNorm1d(embed_size)
                    )
                )
    def init_weights(self):
        """Initialize the weight of network""" 
        def xavier_init(m):
            if type(m) == nn.ConvTranspose1d:
                xavier_normal(m.weight)
        for layer in self.layers:
            layer.apply(xavier_init)

    def forward(self, x):
        out = self.layers[0](x)
        #print(out.size())
        for i in range(len(self.layers) - 1):
            out = self.layers[i + 1](out)
            #print(out.size())
        out = nn.functional.normalize(out,dim=1)
        return out





 
