import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from Attention import Attention


class Interacted_User_Aggregation(nn.Module):
    

    def __init__(self, features, u2e, embed_dim, cuda="cpu"):
        super(Interacted_User_Aggregation, self).__init__()

        self.features = features
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, tmp_neighs):
        embed_matrix = torch.empty(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        for i in range(len(nodes)):
            tmp_adj_M = tmp_neighs[i]
            num_neighs = len(tmp_adj_M)
            # 
            e_u = self.u2e.weight[list(tmp_adj_M)]  

            u_rep = self.u2e.weight[nodes[i]]

            att_w = self.att(e_u, u_rep, num_neighs)
            att_history = torch.mm(e_u.t(), att_w).t()
            embed_matrix[i] = att_history
        to_feats = embed_matrix

        return to_feats