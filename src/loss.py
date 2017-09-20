import torch
from torch.autograd import Variable
from torch import nn

class SentenceLoss(nn.Module):
    """ Loss for stence representation match"""
    def __init(self, embedding):
        super(SentenceLoss, self).__init__()

    def forward(self, input, target, embedding,tau, vocab_size):
        up = torch.exp(1/tau*torch.sum(input*target, dim=1))
        down = []
        for item in input:
             down.append(torch.exp(
                            torch.mm(embedding, item.data)
                        ))
        return up,down
