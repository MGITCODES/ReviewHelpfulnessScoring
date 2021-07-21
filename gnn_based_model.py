import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import pickle
import numpy as np
import time
import random
import datetime
import argparse
import os
from torch.nn import init
from torch.autograd import Variable
from collections import defaultdict
from user_review_encoder import U_R_Encoder
from user_review_aggregation_unit import U_R_Aggregation
from interacted_user_review_encoder import Interact_U_R_Encoder
from interacted_user_aggregation_unit import Interacted_User_Aggregation
from interacted_review_aggregation_unit import Interacted_Review_Aggregation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt


class GnnModel(nn.Module):

    def __init__(self, enc_u, enc_r_history, rating2e):
        super(GnnModel, self).__init__()
        self.enc_u = enc_u
        self.enc_r_history = enc_r_history
        self.embed_dim = enc_u.embed_dim

        self.w_uo1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uo2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ro1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ro2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, 16)
        self.w_ur3 = nn.Linear(16, 1)
        self.rating2e = rating2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_r):
        embeds_u = self.enc_u(nodes_u)
        embeds_r = self.enc_r_history(nodes_r)

        x_u = F.relu(self.bn1(self.w_uo1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_uo2(x_u)
        x_r = F.relu(self.bn2(self.w_vr1(embeds_r)))
        x_r = F.dropout(x_r, training=self.training)
        x_r = self.w_vr2(x_r)

        x_ur = torch.cat((x_u, x_r), 1)
        x = F.relu(self.bn3(self.w_ur1(x_ur)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_ur2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_ur3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_r, labels_list):
        scores = self.forward(nodes_u, nodes_r)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_r, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_r.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_r, tmp_target in test_loader:
            test_u, test_r, tmp_target = test_u.to(device), test_r.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_r)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='User specific review scoring: GnnModel model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    dir_data = './data/Ciao_dataset'

    path_data = dir_data + ".pickle"
    data_file = open(path_data, 'rb')
    history_u_lists, history_ura_lists, history_r_lists, history_rra_lists, train_u, train_r, train_rating, test_u, test_r, test_rating, adj_lists, ratings_list = pickle.load(
        data_file)
   

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_r),
                                              torch.FloatTensor(train_rating))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_r),
                                             torch.FloatTensor(test_rating))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_reviews = history_r_lists.__len__()
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    r2e = nn.Embedding(num_reviews, embed_dim).to(device)
    rating2e = nn.Embedding(num_ratings, embed_dim).to(device)

    agg_u_history = U_R_Aggregation(r2e, rating2e, u2e, embed_dim, cuda=device, ur=True)
    enc_u_history = U_R_Encoder(u2e, embed_dim, history_u_lists, history_ura_lists, agg_u_history, cuda=device, ur=True)
   
    agg_u_inter = Interacted_User_Aggregation(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    enc_u = Interact_U_R_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, adj_lists, agg_u_inter,
                           base_model=enc_u_history, cuda=device)
    agg_r_inter = Interacted_Review_Aggregation(lambda nodes: enc_r_history(nodes).t(), r2e, embed_dim, cuda=device)
    enc_r = Interact_U_R_Encoder(lambda nodes: enc_r_history(nodes).t(), embed_dim, adj_lists, agg_r_inter, cuda=device)

    
    agg_r_history = U_R_Aggregation(r2e, rating2e, u2e, embed_dim, cuda=device, uv=False)
    enc_r_history = U_R_Encoder(r2e, embed_dim, history_r_lists, history_rra_lists, agg_r_history, cuda=device, uv=False)

    # model
    gnnmodel = GnnModel(enc_u, enc_r_history, rating2e).to(device)
    optimizer = torch.optim.RMSprop(GnnModel.parameters(), lr=args.lr, alpha=0.9)

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, args.epochs + 1):

        train(gnnmodel, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        expected_rmse, mae = test(gnnmodel, device, test_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            break


if __name__ == "__main__":
    main()