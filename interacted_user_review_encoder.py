import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Interact_U_R_Encoder(nn.module):
	
	def __init__(self, features, embed_dim, int_u_r_lists, aggre, cuda="cpu"):
		
		super(Interact_U_R_Encoder, self).__init__()
		
		self.features = features
		self.int_u_r_lists = int_u_r_lists
		self.aggre = aggre
		self.embed_dim = embed_dim
		self.device = cuda
		# 可能有问题
		self.linear1 = nn.Linear(2*self.embed_dim, self.embed_dim)
		
		
	def forward(self, nodes):
		
		tmp_neighbors = []
		for node in nodes:
			tmp_neighbors.append(self.int_u_r_lists[int(node)])
			
		neighbor_feats = self.aggre.forward(nodes, tmp_neighbors)
			
		self_feats = self.features.weight[nodes]
			
		combined = torch.cat([self_feats, neighbor_feats],dim=1)
		combined = F.relu(self.linear1(combined))
			
		return combined