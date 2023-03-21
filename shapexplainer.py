import random
import time
from copy import deepcopy
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.special
import torch
import torch_geometric
from tqdm import tqdm
from sklearn.linear_model import (LassoLars, Lasso,
                                  LinearRegression, Ridge)
from sklearn.metrics import r2_score
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GNNExplainer as GNNE
from torch_geometric.nn import MessagePassing

from src.models import LinearRegressionModel
from src.plots import (denoise_graph, k_hop_subgraph, log_graph,
                       visualize_subgraph, custom_to_networkx)

class SHAPExplainer():

    def __init__(self, data, model, gpu=False):
        self.model = model
        self.data = data
        self.gpu = gpu
        self.neighbours = None
        self.F = None  
        self.M = None  
        self.base_values = []
        self.model.eval()

    def explain(self, node_indexes=[0], hops=2, num_samples=10, info=True, multiclass=False,
                fullempty=None, S=3, args_hv='compute_pred', args_feat='Expectation',
                args_coal='Smarter', args_g='WLS', regu=None, visual=False):
        phi_list = []
        for node_index in node_indexes:
            with torch.no_grad():
                true_conf, true_pred = self.model(self.data.x, self.data.edge_index).exp()[node_index].max(dim=0)

            self.neighbours, _, _, edge_mask =\
                torch_geometric.utils.k_hop_subgraph(node_idx=node_index, num_hops=hops, edge_index=self.data.edge_index)
            
            one_hop_neighbours, _, _, _ =\
                torch_geometric.utils.k_hop_subgraph(node_idx=node_index, num_hops=1, edge_index=self.data.edge_index)

            self.neighbours = self.neighbours[self.neighbours != node_index]
            D = self.neighbours.shape[0]

            if args_hv == 'compute_pred_subgraph':
                feat_idx, discarded_feat_idx = self.feature_selection_subgraph(node_index, args_feat)
                    args_hv = 'Smarter' 
            else: 
                feat_idx, discarded_feat_idx = self.feature_selection(node_index, args_feat)

            if regu==1 or D==0: 
                D=0
            if regu==0 or self.F==0:
                self.F=0
            self.M = self.F+D

            args_K = S

            z_, weights = self.mask_generation(num_samples, args_coal, args_K, D, info, regu)

            if fullempty:
                weights[weights == 1000] = 0

            fz = eval('self.' + args_hv)(node_index, num_samples, D, z_, feat_idx, one_hop_neighbours, args_K, args_feat,
                                        discarded_feat_idx, multiclass, true_pred)

            phi, base_value = eval('self.' + args_g)(z_, weights, fz, multiclass, info)

            if type(regu) == int and not multiclass:
                expl = (true_conf.cpu() - base_value).detach().numpy()
                phi[:self.F] = (regu * expl / sum(phi[:self.F])) * phi[:self.F]
                phi[self.F:] = ((1-regu) * expl / sum(phi[self.F:])) * phi[self.F:]

            if info:
                self.print_info(D, node_index, phi, feat_idx, true_pred, true_conf, base_value, multiclass)

            if visual:
                self.visual(edge_mask, node_index, phi, true_pred, hops, multiclass)

            phi_list.append(phi)
            self.base_values.append(base_value)

        return phi_list

    def explain_graphs(self, graph_indices=[0], hops=2, num_samples=10, info=True, multiclass=False,
                       fullempty=None, S=3, args_hv='compute_pred', args_feat='Expectation',
                       args_coal='Smarter', args_g='WLS', regu=None, visual=False):
        phi_list = []
        for graph_index in graph_indices:
            with torch.no_grad():
                true_conf, true_pred = self.model(self.data.x,
                    self.data.edge_index).exp()[graph_index,:].max(dim=0)

            self.neighbours = list(
                range(int(self.data.edge_index.shape[1] - np.sum(np.diag(self.data.edge_index[graph_index])))))
            D = len(self.neighbours)

            self.F = 0
            self.M = self.F+D

            args_K = S

            z_, weights = self.mask_generation(num_samples, args_coal, args_K, D, info, regu)
            
            if fullempty:
                weights[(weights == 1000).nonzero()] = 0

            fz = self.graph_classification(graph_index, num_samples, D, z_, args_K, args_feat, true_pred)

            phi, base_value = eval('self.' + args_g)(z_, weights, fz, multiclass, info)

            phi_list.append(phi)
            self.base_values.append(base_value)

            return phi_list

    def feature_selection(self, node_index, args_feat):
        discarded_feat_idx = []
        std = self.data.x.std(axis=0)
        mean = self.data.x.mean(axis=0)
        mean_subgraph = self.data.x[node_index, :]
        mean_subgraph = torch.where(mean_subgraph >= mean - 0.25*std, mean_subgraph, torch.ones_like(mean_subgraph)*100)
        mean_subgraph = torch.where(mean_subgraph <= mean + 0.25*std, mean_subgraph, torch.ones_like(mean_subgraph)*100)
        feat_idx = (mean_subgraph == 100).nonzero()
        discarded_feat_idx = (mean_subgraph != 100).nonzero()
        self.F = feat_idx.shape[0]
        return feat_idx, discarded_feat_idx

    def feature_selection_subgraph(self, node_index, args_feat):
        discarded_feat_idx = []
        std = self.data.x.std(axis=0)
        mean = self.data.x.mean(axis=0)
        mean_subgraph = torch.mean(self.data.x[self.neighbours, :], dim=0)
        mean_subgraph = torch.where(mean_subgraph >= mean - 0.25*std, mean_subgraph, torch.ones_like(mean_subgraph)*100)
        mean_subgraph = torch.where(mean_subgraph <= mean + 0.25*std, mean_subgraph, torch.ones_like(mean_subgraph)*100)
        feat_idx = (mean_subgraph == 100).nonzero()
        discarded_feat_idx = (mean_subgraph != 100).nonzero()
        self.F = feat_idx.shape[0]
        del mean, mean_subgraph, std
        return feat_idx, discarded_feat_idx

    def mask_generation(self, num_samples, args_coal, args_K, D, info, regu):
        num_samples = min(10000, 2**self.M)
        z_ = eval('self.' + args_coal)(num_samples, args_K, regu)
        z_ = z_[torch.randperm(z_.size()[0])]
        s = (z_ != 0).sum(dim=1)
        weights = self.shapley_kernel(s, self.M)
        return z_, weights

    def SHAP(self, s, M):
        shapley = []

        for i in range(s.shape[0]):
            a = s[i].item()
            if a == 0 or a == M:
                shapley.append(1000)
            elif scipy.special.binom(M, a) == float('+inf'):
                shapley.append(1/ (M**2))
            else:
                shapley.append(
                    (M-1)/(scipy.special.binom(M, a)*a*(M-a)))

        shapley = np.array(shapley)
        shapley = np.where(shapley<1.0e-40, 1.0e-40,shapley)
        return torch.tensor(shapley)

    def WLS(self, z_, weights, fz, multiclass, info):
        z_ = torch.cat([z_, torch.ones(z_.shape[0], 1)], dim=1)
        tmp = np.dot(np.dot(z_.T, np.diag(weights)), z_)
        tmp = np.linalg.inv(tmp + np.diag(10**(-5) * np.random.randn(tmp.shape[1])))
        phi = np.dot(tmp, np.dot(np.dot(z_.T, np.diag(weights)), fz.detach().numpy()))
        y_pred = z_.detach().numpy() @ phi
        return phi[:-1], phi[-1]

    def WLR(self, z_, weights, fz, multiclass, info):
        if multiclass:
            our_model = LinearRegressionModel(z_.shape[1], self.data.num_classes)
        else:
            our_model = LinearRegressionModel(z_.shape[1], 1)
        our_model.train()

        def weighted_mse_loss(input, target, weight):
            return (weight * (input - target) ** 2).mean()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(our_model.parameters(), lr=0.001)

        train = torch.utils.data.TensorDataset(z_, fz, weights)
        train_loader = torch.utils.data.DataLoader(train, batch_size=1)

        for epoch in range(100):
            av_loss = []
            for batch_idx, (dat, target, w) in enumerate(train_loader):
                x, y, w = Variable(dat), Variable(target), Variable(w)
                pred_y = our_model(x)
                loss = weighted_mse_loss(pred_y, y, w)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                av_loss.append(loss.item())
            if epoch % 10 ==0 and info:
                print('av loss epoch: ', np.mean(av_loss))

        our_model.eval()
        with torch.no_grad():
            pred = our_model(z_)
        if info:
            print('weighted r2 score: ', r2_score(pred, fz, multioutput='variance_weighted'))
            if multiclass:
                print(r2_score(pred, fz, multioutput='raw_values'))
            print('r2 score: ', r2_score(pred, fz, weights))

        phi, base_value = [param.T for _,param in our_model.named_parameters()]
        phi = np.squeeze(phi, axis=1)
        return phi.detach().numpy().astype('float64'), base_value.detach().numpy().astype('float64')

    def visual(self, edge_mask, node_index, phi, predicted_class, hops, multiclass):
        if multiclass:
            phi = torch.tensor(phi[predicted_class, :])
        else:
            phi = torch.from_numpy(phi).float()

        mask = edge_mask.int().float()

        one_hop_nei, _, _, _ = torch_geometric.utils.k_hop_subgraph(node_index, 1, self.data.edge_index, relabel_nodes=True,num_nodes=None)

        for i, nei in enumerate(self.neighbours):
            list_indexes = (self.data.edge_index[0, :] == nei).nonzero()
            for idx in list_indexes:
                if nei in one_hop_nei:
                    if self.data.edge_index[1, idx] in one_hop_nei:
                        mask[idx] = phi[self.F + i]
                elif mask[idx] == 1:
                    mask[idx] = phi[self.F + i]

        mask[mask == 1] = 0
        mask = torch.abs(mask)
        mask = mask / sum(mask)

        ax, G = visualize_subgraph(self.model, node_index, self.data.edge_index, mask, hops, y=self.data.y, threshold=None)

        plt.savefig('results/GS1_{}_{}_{}'.format(self.data.name, self.model.__class__.__name__, node_index), bbox_inches='tight')
