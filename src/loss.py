import torch
from torch.autograd import Variable
from torch import nn

class DeconvCNNLoss(nn.Module):
    def __init__(self, vocab_size, embed, tau=0.01):
        super(DecovCNNLoss, self).__init__()
        self.dist_helper = nn.Linear(embed,vocab_size,bias=False)
        self.tau = tau

    def forward(self,embed_EN, embed_DE, embed_M):
        self.dist_helper.weight.data = embed_M
        distance = self.dist_helper(embed_DE.transpose(1,2))
        up = torch.exp(1/self.tau*torch.sum(embed_EN*embed_DE, dim=1))
        down = torch.sum(torch.exp(1/self.tau*distance), dim=2)
        loss = torch.sum(up/down)
        return loss
