import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention


class U_R_Aggregation(nn.Module):

    def __init__(self, u2e, r2e, rating2e, embed_dim, cuda="cpu", ur=True):
        super(U_R_Aggregation, self).__init__()
        self.ur = ur
        self.u2e = u2e
        self.r2e = r2e
        self.rangti2e = rating2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, ur_history_lists, rating_history_lists):

        embed_matrix = torch.empty(len(ur_history_lists), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(ur_history_lists)):
            history = ur_history_lists[i]
            num_histroy_review = len(history)
            tmp_label = rating_history_lists[i]

            if self.ur == True:
                
                e_ur = self.r2e.weight[history]
                ur_rep = self.u2e.weight[nodes[i]]
            else:
                
                e_ur = self.u2e.weight[history]
                ur_rep = self.r2e.weight[nodes[i]]

            e_rating = self.rating2e.weight[tmp_label]
            x = torch.cat((e_ur, e_rating), 1)
            x = F.relu(self.w_r1(x))
            o_history = F.relu(self.w_r2(x))

            att_w = self.att(o_history, ur_rep, num_histroy_review)
            att_history = torch.mm(o_history.t(), att_w)
            att_history = att_history.t()

            embed_matrix[i] = att_history
        to_feats = embed_matrix
        return to_feats