import torch
from torch import nn
import torch.nn.functional as F
import pdb
import numpy as np
from utils.utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LinearRegression(torch.nn.Module):
    # TODO: add more stuff?
    def __init__(self, input_width,
            n_output):
        super(LinearRegression, self).__init__()

        self.final_layer = nn.Sequential(
            nn.Linear(input_width, n_output, bias=True),
            nn.Sigmoid()
        ).to(device)

    def forward(self, x):
        output = x
        output = self.final_layer(output)
        return output

class CostModelNet(torch.nn.Module):
    def __init__(self, input_width, hidden_width_multiple,
            n_output, num_hidden_layers=1, hidden_layer_size=None):
        super(CostModelNet, self).__init__()
        if hidden_layer_size is None:
            n_hidden = int(input_width * hidden_width_multiple)
        else:
            n_hidden = hidden_layer_size

        self.layers = []
        self.layer1 = nn.Sequential(
            nn.Linear(input_width, n_hidden, bias=True),
            nn.LeakyReLU()
        ).to(device)
        self.layers.append(self.layer1)

        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(n_hidden, n_hidden, bias=True),
                nn.LeakyReLU()
            ).to(device)
            self.layers.append(layer)

        self.final_layer = nn.Sequential(
            nn.Linear(n_hidden, n_output, bias=True),
            # nn.Sigmoid()
        ).to(device)
        self.layers.append(self.final_layer)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

class SimpleRegression(torch.nn.Module):
    # TODO: add more stuff?
    def __init__(self, input_width, hidden_width_multiple,
            n_output, num_hidden_layers=1, hidden_layer_size=None,
            use_batch_norm=False):
        super(SimpleRegression, self).__init__()
        if hidden_layer_size is None:
            n_hidden = int(input_width * hidden_width_multiple)
        else:
            n_hidden = hidden_layer_size

        # self.layers = []
        self.layers = nn.ModuleList()

        if use_batch_norm:
            layer1 = nn.Sequential(
                nn.Linear(input_width, n_hidden, bias=True),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU()
            ).to(device)
        else:
            layer1 = nn.Sequential(
                nn.Linear(input_width, n_hidden, bias=True),
                nn.ReLU()
            ).to(device)

        self.layers.append(layer1)

        for i in range(0,num_hidden_layers-1,1):
            if use_batch_norm:
                layer = nn.Sequential(
                    nn.Linear(n_hidden, n_hidden, bias=True),
                    nn.BatchNorm1d(n_hidden),
                    nn.ReLU()
                ).to(device)
                self.layers.append(layer)
            else:
                layer = nn.Sequential(
                    nn.Linear(n_hidden, n_hidden, bias=True),
                    nn.ReLU()
                ).to(device)

            self.layers.append(layer)

        final_layer = nn.Sequential(
            nn.Linear(n_hidden, n_output, bias=True),
            nn.Sigmoid()
        ).to(device)
        self.layers.append(final_layer)

    def compute_grads(self):
        wts = []
        mean_wts = []
        # for layer in self.layers:
            # wts.append(layer[0].weight.grad)

        # for i,wt in enumerate(wts):
            # mean_wts.append(np.mean(np.abs(wt.detach().numpy())))

        return mean_wts

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output

class PaddedMSCN(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, flow_feats,
            hid_units, dropout=0.0, max_hid=None, num_hidden_layers=2):
        print("device: {}".format(device))
        super(PaddedMSCN, self).__init__()
        print("flow feats: ", flow_feats)
        self.flow_feats = flow_feats

        self.sample_mlp1 = nn.Linear(sample_feats, hid_units).to(device)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units).to(device)

        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units).to(device)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units).to(device)

        self.join_mlp1 = nn.Linear(join_feats, hid_units).to(device)
        self.join_mlp2 = nn.Linear(hid_units, hid_units).to(device)

        if flow_feats > 0:
            self.flow_mlp1 = nn.Linear(flow_feats, hid_units).to(device)
            self.flow_mlp2 = nn.Linear(hid_units, hid_units).to(device)
            self.out_mlp1 = nn.Linear(hid_units * 4, hid_units).to(device)
        else:
            self.out_mlp1 = nn.Linear(hid_units * 3, hid_units).to(device)

        self.out_mlp2 = nn.Linear(hid_units, 1).to(device)

    def forward(self, samples, predicates, joins, flows,
                    sample_mask, predicate_mask, join_mask):
        '''
        #TODO: describe shapes
        '''
        samples = samples.to(device, non_blocking=True)
        predicates = predicates.to(device, non_blocking=True)
        joins = joins.to(device, non_blocking=True)
        sample_mask = sample_mask.to(device, non_blocking=True)
        predicate_mask = predicate_mask.to(device, non_blocking=True)
        join_mask = join_mask.to(device, non_blocking=True)

        if self.flow_feats:
            flows = flows.to(device, non_blocking=True)
            hid_flow = F.relu(self.flow_mlp1(flows))
            hid_flow = F.relu(self.flow_mlp2(hid_flow))

        hid_sample = F.relu(self.sample_mlp1(samples))
        hid_sample = F.relu(self.sample_mlp2(hid_sample))
        hid_sample = hid_sample * sample_mask
        hid_sample = torch.sum(hid_sample, dim=1, keepdim=False)
        sample_norm = sample_mask.sum(1, keepdim=False)
        hid_sample = hid_sample / sample_norm

        hid_predicate = F.relu(self.predicate_mlp1(predicates))
        hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
        hid_predicate = hid_predicate * predicate_mask
        hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
        predicate_norm = predicate_mask.sum(1, keepdim=False)
        hid_predicate = hid_predicate / predicate_norm

        hid_join = F.relu(self.join_mlp1(joins))
        hid_join = F.relu(self.join_mlp2(hid_join))
        hid_join = hid_join * join_mask
        hid_join = torch.sum(hid_join, dim=1, keepdim=False)
        join_norm = join_mask.sum(1, keepdim=False)
        hid_join = hid_join / join_norm

        assert hid_sample.shape == hid_predicate.shape == hid_join.shape
        hid_sample = hid_sample.squeeze()
        hid_predicate = hid_predicate.squeeze()
        hid_join = hid_join.squeeze()
        if self.flow_feats:
            hid = torch.cat((hid_sample, hid_predicate, hid_join, hid_flow), 1)
        else:
            hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)

        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out

    def compute_grads(self):
        wts = []
        # wts.append(self.sample_mlp1[0].weight.grad)
        # wts.append(self.sample_mlp2[0].weight.grad)
        # wts.append(self.predicate_mlp1[0].weight.grad)
        # wts.append(self.predicate_mlp2[0].weight.grad)
        # wts.append(self.join_mlp1[0].weight.grad)
        # wts.append(self.join_mlp2[0].weight.grad)

        # wts.append(self.out_mlp1[0].weight.grad)
        # wts.append(self.out_mlp2[0].weight.grad)

        # if self.flow_feats:
            # wts.append(self.flow_mlp1[0].weight.grad)
            # wts.append(self.flow_mlp2[0].weight.grad)

        mean_wts = []
        for i,wt in enumerate(wts):
            mean_wts.append(np.mean(np.abs(wt.detach().numpy())))

        return mean_wts

# MSCN model, kipf et al.
class MSCN(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, flow_feats,
            hid_units, dropout=0.0, max_hid=None, num_hidden_layers=2):
        super(MSCN, self).__init__()
        self.sample_mlp1 = nn.Linear(sample_feats, hid_units)
        self.sample_mlp2 = nn.Linear(hid_units, hid_units)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units)
        self.predicate_mlp2 = nn.Linear(hid_units, hid_units)
        self.join_mlp1 = nn.Linear(join_feats, hid_units)
        self.join_mlp2 = nn.Linear(hid_units, hid_units)
        self.out_mlp1 = nn.Linear(hid_units * 3, hid_units)
        self.out_mlp2 = nn.Linear(hid_units, 1)

    def forward(self, samples, predicates, joins, flows):
        '''
        #TODO: describe shapes
        '''
        # samples = torch.stack(samples)
        # joins = torch.stack(joins)
        batch_hid_samples = []
        for cur_sample in samples:
            sample = F.relu(self.sample_mlp1(cur_sample))
            sample = F.relu(self.sample_mlp2(sample))
            sample = torch.sum(sample, dim=0, keepdim=False)
            sample = sample / cur_sample.shape[0]
            batch_hid_samples.append(sample)
        hid_sample = torch.stack(batch_hid_samples)

        # going to pass each batch separately since they have different shapes
        # (number of predicates will be different in each case)
        # want hid_predicate of shape batch x num_predicate_feats
        batch_hid_preds = []
        for cur_pred in predicates:
            pred = F.relu(self.predicate_mlp1(cur_pred))
            pred = F.relu(self.predicate_mlp2(pred))
            # avg to create single output
            pred = torch.sum(pred, dim=0, keepdim=False)
            pred = pred / cur_pred.shape[0]
            batch_hid_preds.append(pred)

        hid_predicate = torch.stack(batch_hid_preds)

        # hid_join = F.relu(self.join_mlp1(joins))
        # hid_join = F.relu(self.join_mlp2(hid_join))

        batch_hid_joins = []
        for cur_join in joins:
            join = F.relu(self.join_mlp1(cur_join))
            join = F.relu(self.join_mlp2(join))
            join = torch.sum(join, dim=0, keepdim=False)
            join = join / cur_join.shape[0]
            batch_hid_joins.append(join)
        hid_join = torch.stack(batch_hid_joins)

        hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)
        hid = F.relu(self.out_mlp1(hid))
        out = torch.sigmoid(self.out_mlp2(hid))
        return out

# MVCN?
class SetConv(nn.Module):
    def __init__(self, sample_feats, predicate_feats, join_feats, flow_feats,
            hid_units, dropout=0.0, max_hid=None, num_hidden_layers=2):
        # super(SetConv, self).__init__()
        super().__init__()
        print("initializing mscn with {} hidden layers".format(num_hidden_layers))
        self.dropout = dropout
        self.flow_feats = flow_feats
        # doesn't really make sense to have this be bigger...
        sample_hid = hid_units
        if max_hid is not None:
            sample_hid = min(hid_units, max_hid)

        self.sample_mlps = nn.ModuleList()
        sample_mlp1 = nn.Sequential(
            nn.Linear(sample_feats, sample_hid),
            nn.ReLU()
        ).to(device)
        self.sample_mlps.append(sample_mlp1)

        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(sample_hid, sample_hid),
                nn.ReLU()
            ).to(device)
            self.sample_mlps.append(layer)

        # self.sample_mlp1 = nn.Sequential(
            # nn.Linear(sample_feats, sample_hid),
            # nn.ReLU()
        # ).to(device)

        # self.sample_mlp2 = nn.Sequential(
            # nn.Linear(sample_hid, sample_hid),
            # nn.ReLU()
        # ).to(device)

        self.predicate_mlps = nn.ModuleList()
        self.predicate_mlps.append(nn.Sequential(
            nn.Linear(predicate_feats, hid_units),
            nn.ReLU()
        ).to(device))
        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(hid_units, hid_units),
                nn.ReLU()
            ).to(device)
            self.predicate_mlps.append(layer)

        # self.predicate_mlp1 = nn.Sequential(
            # nn.Linear(predicate_feats, hid_units),
            # nn.ReLU()
        # ).to(device)

        # self.predicate_mlp2 = nn.Sequential(
            # nn.Linear(hid_units, hid_units),
            # nn.ReLU()
        # ).to(device)

        if max_hid is not None:
            join_hid = min(hid_units, max_hid)
        else:
            join_hid = hid_units

        self.join_mlps  = nn.ModuleList()
        self.join_mlps.append(nn.Sequential(
            nn.Linear(join_feats, join_hid),
            nn.ReLU()
        ).to(device))

        for i in range(0,num_hidden_layers-1,1):
            layer = nn.Sequential(
                nn.Linear(join_hid, join_hid),
                nn.ReLU()
            ).to(device)
            self.join_mlps.append(layer)

        # self.join_mlp1 = nn.Sequential(
            # nn.Linear(join_feats, join_hid),
            # nn.ReLU()
        # ).to(device)

        # self.join_mlp2 = nn.Sequential(
            # nn.Linear(join_hid, join_hid),
            # nn.ReLU()
        # ).to(device)

        if flow_feats > 0:
            flow_hid = hid_units
            self.flow_mlps  = nn.ModuleList()
            self.flow_mlps.append(nn.Sequential(
                nn.Linear(self.flow_feats, flow_hid),
                nn.ReLU()
            ).to(device))
            for i in range(0,num_hidden_layers-1,1):
                layer = nn.Sequential(
                    nn.Linear(flow_hid, flow_hid),
                    nn.ReLU()
                ).to(device)
                self.flow_mlps.append(layer)

            # self.flow_mlp1 = nn.Sequential(
                # nn.Linear(self.flow_feats, flow_hid),
                # nn.ReLU()
            # ).to(device)

            # self.flow_mlp2 = nn.Sequential(
                # nn.Linear(flow_hid, flow_hid),
                # nn.ReLU()
            # ).to(device)

        total_hid = sample_hid + join_hid + hid_units

        if flow_feats:
            total_hid += flow_hid

        self.out_mlp1 = nn.Sequential(
                nn.Linear(total_hid, hid_units),
                nn.ReLU()
        ).to(device)

        self.out_mlp2 = nn.Sequential(
                nn.Linear(hid_units, 1),
                nn.Sigmoid()
        ).to(device)

        # self.drop_layer = nn.Dropout(self.dropout)

    def compute_grads(self):
        wts = []
        # wts.append(self.sample_mlp1[0].weight.grad)
        # wts.append(self.sample_mlp2[0].weight.grad)
        # wts.append(self.predicate_mlp1[0].weight.grad)
        # wts.append(self.predicate_mlp2[0].weight.grad)
        # wts.append(self.join_mlp1[0].weight.grad)
        # wts.append(self.join_mlp2[0].weight.grad)

        # wts.append(self.out_mlp1[0].weight.grad)
        # wts.append(self.out_mlp2[0].weight.grad)

        # if self.flow_feats:
            # wts.append(self.flow_mlp1[0].weight.grad)
            # wts.append(self.flow_mlp2[0].weight.grad)

        mean_wts = []
        for i,wt in enumerate(wts):
            mean_wts.append(np.mean(np.abs(wt.detach().numpy())))

        return mean_wts

    def forward(self, samples, predicates, joins,
            flows):

        # hid_sample = self.sample_mlp1(samples)
        # hid_sample = self.drop_layer(hid_sample)
        # hid_sample = self.sample_mlp2(hid_sample)
        hid_sample = samples
        for layer in self.sample_mlps:
            hid_sample = layer(hid_sample)

        hid_predicate = predicates
        for layer in self.predicate_mlps:
            hid_predicate = layer(hid_predicate)
            # TODO: add dropout

        # hid_predicate = self.predicate_mlp1(predicates)
        # hid_predicate = self.drop_layer(hid_predicate)
        # hid_predicate = self.predicate_mlp2(hid_predicate)

        hid_join = joins
        for layer in self.join_mlps:
            hid_join = layer(hid_join)

        # hid_join = self.join_mlp1(joins)
        # hid_join = self.join_mlp2(hid_join)

        if self.flow_feats:
            # hid_flow = self.flow_mlp1(flows)
            # hid_flow = self.flow_mlp2(hid_flow)
            hid_flow = flows
            for layer in self.flow_mlps:
                hid_flow = layer(hid_flow)

            hid = torch.cat((hid_sample, hid_predicate, hid_join, hid_flow), 1)
        else:
            hid = torch.cat((hid_sample, hid_predicate, hid_join), 1)

        hid = self.out_mlp1(hid)
        out = self.out_mlp2(hid)

        return out
