import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import os
from utils.graph_utils import GraphSampler
from torch.autograd import Variable


def train_and_val(model, data, num_epochs, lr, wd, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    train_loss_values, train_acc_values = [], []
    val_loss_values, val_acc_values = [], []

    best = np.inf
    bad_counter = 0

    for epoch in tqdm(range(num_epochs), desc='Training', leave=False):
        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        train_loss = F.nll_loss(
            output[data.train_mask], data.y[data.train_mask])
            
        train_acc = accuracy(output[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        
        train_loss_values.append(train_loss.item())
        train_acc_values.append(train_acc.item())

        val_loss, val_acc = evaluate(data, model, data.val_mask)
        val_loss_values.append(val_loss.item())
        val_acc_values.append(val_acc.item())

def evaluate(data, model, mask):
    model.eval()
    with torch.no_grad():
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[mask], data.y[mask])
        acc = accuracy(output[mask], data.y[mask])

    return loss, acc

def accuracy(output, labels):
    _, pred = output.max(dim=1)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_syn(data, model, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.num_epochs):
        total_loss = 0
        model.train()
        opt.zero_grad()

        pred = model(data.x, data.edge_index)
        pred = pred[data.train_mask]
        label = data.y[data.train_mask]

        loss = F.nll_loss(pred, label)
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.clip)
        opt.step()
        total_loss += loss.item() * 1
        
        if epoch % 10 == 0:
            train_acc = test(data, model, data.train_mask)
            val_acc = test(data, model, data.val_mask)
            print("Epoch {}. Loss: {:.4f}. Train accuracy: {:.4f}. Val accuracy: {:.4f}".format(epoch, total_loss, train_acc, val_acc))
    total_loss = total_loss / data.x.shape[0]

def test(data, model, mask):
    model.eval()
    correct = 0

    with torch.no_grad():
        pred = model(data.x, data.edge_index)
        pred = pred.argmax(dim=1)
        pred = pred[mask]
        label = data.y[mask]
        correct += pred.eq(label).sum().item()

    total = (mask == True).nonzero().shape[0]
    return correct / total

def train_gc(data, model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset_sampler = GraphSampler(data)
    train_dataset_loader = torch.utils.data.DataLoader(dataset_sampler,batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.num_epochs):
        avg_loss = 0.0
        model.train()
        for batch_idx, df in enumerate(train_dataset_loader):
            optimizer.zero_grad()
            y_pred = model(df["feats"], df["adj"])
            loss = F.nll_loss(y_pred, df['label'])
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), args.clip)
            optimizer.step()
            avg_loss += loss

        avg_loss /= batch_idx + 1
        if epoch % 10 == 0:
            train_acc = test(data, model, data.train_mask)
            val_acc = test(data, model, data.val_mask)
            print("Epoch {}. Loss: {:.4f}. Train accuracy: {:.4f}. Val accuracy: {:.4f}".format(epoch, avg_loss, train_acc, val_acc))