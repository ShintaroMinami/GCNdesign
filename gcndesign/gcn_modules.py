import torch
import torch.nn as nn
from einops import rearrange
import math

##  Custom Normalization
class ExChannelNorm(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, **kwargs)
    def forward(self, x):
        shape_org = x.shape
        x = self.norm(rearrange(x, 'b ... c -> (b ...) c'))
        return x.view(shape_org)


##  ResBlock with BatchNormalization
class ResBlock(nn.Module):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            dropout: float=0.2,
        ):
        super().__init__()
        self.layer1 = nn.Sequential(
            ExChannelNorm(d_in, affine=True),
            nn.ReLU(),
            nn.Linear(d_in, d_out)
        )
        self.layer2 = nn.Sequential(
            ExChannelNorm(d_out, affine=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out)
        )
        self.shortcut = nn.Sequential()
        if d_in != d_out:
            self.shortcut.add_module('norm', ExChannelNorm(d_in, affine=True))
            self.shortcut.add_module('ff', nn.Linear(d_in, d_out))
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out += self.shortcut(x)
        return out


## Graph Convolution Block
class RGCBlock(nn.Module):
    def __init__(
            self,
            d_node_in: int,
            d_node_out: int,
            d_edge_in: int,
            d_edge_out: int,
            d_node_hidden: int=128,
            d_edge_hidden: int=128,
            nlayer_node: int=2,
            nlayer_edge: int=2,
            knn: int=20,
            r_dropout: float=0.2,
        ):
        super().__init__()
        self.nneighbor = knn
        self.nlayer_edge = nlayer_edge
        self.d_node_in = d_node_in
        self.d_node_out = d_node_out
        k_node = d_node_out - d_node_in
        k_edge = d_edge_out - d_edge_in
        #  edge update layer
        if(nlayer_edge > 0):
            self.edgeupdate = nn.Sequential(
                nn.Linear(d_edge_in+d_node_in+d_node_in, d_edge_hidden),
                *[ResBlock(d_edge_hidden, d_edge_hidden, dropout=r_dropout) for _ in range(nlayer_edge-1)],
                ResBlock(d_edge_hidden, k_edge, dropout=r_dropout),
                ExChannelNorm(k_edge, affine=True),
                nn.ReLU(),
            )
        #  graph convolution layer
        self.encoding = nn.Sequential(
            ExChannelNorm(d_edge_out+2*d_node_in, affine=True),
            nn.Linear(d_edge_out+2*d_node_in, d_node_hidden),
            *[ResBlock(d_node_hidden, d_node_hidden, dropout=r_dropout) for _ in range(nlayer_node)],
            ExChannelNorm(d_node_hidden, affine=True),
            nn.ReLU(),
        )
        #  residual update layer
        self.residual = nn.Sequential(
            *[ResBlock(d_node_hidden, d_node_hidden, dropout=r_dropout) for _ in range(nlayer_node-1)],
            ResBlock(d_node_hidden, k_node, dropout=r_dropout),
            ExChannelNorm(k_node, affine=True),
            nn.ReLU(),
        )
    
    def get_target_and_source_nodes(self, node, adjmat):
        b, l, _ = node.shape
        node_expand = node.expand(b, l, l, self.d_node_in)
        nodetrg = node_expand[adjmat, :].reshape(b, l, -1, self.d_node_in)
        nodesrc = node.unsqueeze(2).expand(b, l, self.nneighbor, self.d_node_in)
        return nodetrg, nodesrc

    def forward(self, x, edgevec, adjmat):
        ## edge update ##
        nodetrg, nodesrc = self.get_target_and_source_nodes(x, adjmat)
        # concat node-vec & edge-vec
        if(self.nlayer_edge > 0):
            node_edge_node = torch.cat((nodesrc, edgevec, nodetrg), -1)
            edgevec_residual = self.edgeupdate(node_edge_node)
            edgevec = torch.cat((edgevec, edgevec_residual), -1)
        ## node update ##
        node_edge_node = torch.cat((nodesrc, edgevec, nodetrg), -1)
        # encoding layer
        encoded = self.encoding(node_edge_node)
        # aggregation
        aggregated = encoded.sum(-2) / math.sqrt(self.nneighbor)
        # residual
        residual = self.residual(aggregated)
        # add dense connection
        out = torch.cat((x, residual), -1)
        # return
        return out, edgevec

