import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class U_R_Encoder(nn.module):
	
	def __init__(self, features, embed_dim, ur_history_lists, rating_history_lists, aggre, ur=True, cuda="cpu"):
		super(U_R_Encoder, self).__init__()
		
		self.features = features
		self.ur = ur
		self.ur_history_lists = ur_history_lists
		self.rating_history_lists = rating_history_lists
		self.aggre = aggre
		self.embed_dim = embed_dim
		self.device = cuda
		self.linear1 = nn.Linear(2*self.embed_dim)
		
		
	def forward(self, nodes):
		tmp_ur_history = []
		tmp_rating_history = []
		for node in nodes:
			tmp_ur_history.append(self.ur_history_lists[int(node)])
			tmp_rating_history.append(self.rating_history_lists[int(node)])
			
		neighbor_feats = self.aggre.forward(nodes, tmp_ur_history, tmp_rating_history)
		self_feats = self.features.weight[nodes]
		combined = torch.cat([self_feats, neighbor_feats],dim=1)
		combined = F.relu(self.linear1(combined))
		return combined

