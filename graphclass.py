import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric.nn import GNNExplainer as GNNE

from models import GAT, GCN
from train import accuracy, train_and_val
from utils import *
from data import (add_noise_features, add_noise_neighbours,
                      extract_test_nodes, prepare_data)
from explainers import SHAPExplainer

def graphclass(data, model, args):
    allgraphs = list(range(len(data.selected)))[100:120]
    accuracy = []
    for graph_idx in allgraphs:
        shapexplainer = SHAPExplainer(data, model, args.gpu)
        shap_explainers = shapexplainer.explain_graphs([graph_idx],args.hops,args.num_samples,args.info,
                                                        args.multiclass,args.fullempty,args.S,
                                                        'graph_classification',args.feat,args.coal,
                                                        args.g,regu=0,visual=False)[0]
        idexs = np.nonzero(data.edge_label_lists[graph_idx])[0].tolist()
        inter = [] 
        for i in idexs:
            inter.append(data.edge_lists[graph_idx][i])
        ground_truth = [item for sublist in inter for item in sublist]
        ground_truth = list(set(ground_truth))
        k = len(ground_truth)  # Length gt
        if len(shapexplainer.neighbours) >= k:
            i = 0
            val, indices = torch.topk(torch.tensor(
                shap_explainers.T), k)
            for node in torch.tensor(shapexplainer.neighbours)[indices]:
                if node.item() in ground_truth:
                    i += 1
            accuracy.append(i / k)
            print('acc:', i/k)
            print('indexes', indices)
            print('gt', ground_truth)
    print('Accuracy', accuracy)
    print('Mean accuracy', np.mean(accuracy))
