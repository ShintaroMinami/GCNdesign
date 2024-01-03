import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
from matplotlib import pyplot as plt

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
            d_edge_hidden: int=64,
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
        node_expand = repeat(node, 'b l c -> b x l c', x=l)
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


## Graph Convolution Block
class RGCBlock_simple(nn.Module):
    def __init__(
            self,
            d_node: int=128,
            d_edge: int=128,
            nlayer_node: int=2,
            nlayer_edge: int=2,
            knn: int=20,
            r_dropout: float=0.2,
        ):
        super().__init__()
        self.nneighbor = knn
        self.nlayer_edge = nlayer_edge
        self.d_node = d_node
        self.d_edge = d_edge
        #  edge update layer
        if(nlayer_edge > 0):
            self.edgeupdate = nn.Sequential(
                nn.Linear(d_edge+2*d_node, d_edge),
                *[ResBlock(d_edge, d_edge, dropout=r_dropout) for _ in range(nlayer_edge-1)],
                ResBlock(d_edge, d_edge, dropout=r_dropout),
                ExChannelNorm(d_edge, affine=True),
                nn.ReLU(),
            )
        #  graph convolution layer
        self.encoding = nn.Sequential(
            ExChannelNorm(d_edge+2*d_node, affine=True),
            nn.Linear(d_edge+2*d_node, d_node),
            *[ResBlock(d_node, d_node, dropout=r_dropout) for _ in range(nlayer_node)],
            ExChannelNorm(d_node, affine=True),
            nn.ReLU(),
        )
        #  residual update layer
        self.residual = nn.Sequential(
            *[ResBlock(d_node, d_node, dropout=r_dropout) for _ in range(nlayer_node-1)],
            ResBlock(d_node, d_node, dropout=r_dropout),
            ExChannelNorm(d_node, affine=True),
            nn.ReLU(),
        )
    
    def get_target_and_source_nodes(self, node, adjmat):
        b, l, _ = node.shape
        node_expand = node.expand(b, l, l, self.d_node)
        nodetrg = node_expand[adjmat, :].reshape(b, l, -1, self.d_node)
        nodesrc = node.unsqueeze(2).expand(b, l, self.nneighbor, self.d_node)
        return nodetrg, nodesrc

    def forward(self, x, edgevec, adjmat):
        ## edge update ##
        nodetrg, nodesrc = self.get_target_and_source_nodes(x, adjmat)
        # concat node-vec & edge-vec
        if(self.nlayer_edge > 0):
            node_edge_node = torch.cat((nodesrc, edgevec, nodetrg), -1)
            edgevec_residual = self.edgeupdate(node_edge_node)
            edgevec = edgevec + edgevec_residual#torch.cat((edgevec, edgevec_residual), -1)
        ## node update ##
        node_edge_node = torch.cat((nodesrc, edgevec, nodetrg), -1)
        # encoding layer
        encoded = self.encoding(node_edge_node)
        # aggregation
        aggregated = encoded.sum(-2) / math.sqrt(self.nneighbor)
        # residual
        residual = self.residual(aggregated)
        # add dense connection
        #out = torch.cat((x, residual), -1)
        out = x + residual
        # return
        return out, edgevec


def FeedForward(dim, mult=1., num_layers=2, act=nn.ReLU):
    layers = []
    dim_hidden = dim * mult
    for ind in range(num_layers):
        is_first = ind == 0
        is_last  = ind == (num_layers - 1)
        dim_in   = dim if is_first else dim_hidden
        dim_out  = dim if is_last else dim_hidden
        layers.append(nn.Linear(dim_in, dim_out))
        if is_last:
            continue
        layers.append(act())
    return nn.Sequential(*layers)


## Graph Convolution Block
from rotary_embedding_torch import RotaryEmbedding
from einops.layers.torch import Rearrange
class GTFBlock(nn.Module):
    def __init__(
            self,
            d_node: int=128,
            d_edge: int=64,
            nlayer_node: int=2,
            nlayer_edge: int=2,
            heads: int=8,
            key_dim: int=16,
            value_dim: int=16,
            value_dim_edge: int=8,
            knn: int=20,
            r_dropout: float=0.2,
        ):
        super().__init__()
        self.nneighbor = knn
        self.nlayer_edge = nlayer_edge
        self.d_node = d_node
        self.d_edge = d_edge
        #  edge update layer
        if(nlayer_edge > 0):
            self.edgeupdate = nn.Sequential(
                nn.Linear(d_edge+d_node+d_node, d_edge),
                *[ResBlock(d_edge, d_edge, dropout=r_dropout) for _ in range(nlayer_edge-1)],
                ResBlock(d_edge, d_edge, dropout=r_dropout),
                ExChannelNorm(d_edge, affine=True),
                nn.ReLU(),
            )
        #  graph convolution layer
        #self.attn_norm = nn.LayerNorm(d_node)
        self.attn_norm = ExChannelNorm(d_node, affine=True)
        # num attention contributions
        num_attn_logits = 2
        self.key_dim, self.value_dim, self.heads = key_dim, value_dim, heads
        # qkv projection for attention (normal)
        self.attn_logits_scale = (num_attn_logits * key_dim) ** -0.5
        self.to_q = nn.Linear(d_node, key_dim * heads, bias = False)
        self.to_k = nn.Linear(d_node, key_dim * heads, bias = False)
        self.to_v = nn.Linear(d_node, value_dim * heads, bias = False)
        # rotary embedding
        self.rotary_emb = RotaryEmbedding(dim=key_dim)
        # edge representation projection to attention bias
        self.edge_attn_logits_scale = num_attn_logits ** -0.5
        self.to_edge_attn_bias = nn.Sequential(
            nn.Linear(d_edge, heads),
            Rearrange('b ... h -> (b h) ...')
        )
        self.to_edge_v = nn.Linear(d_edge, value_dim_edge * heads)
        # combine out - node + edge dim
        d_attn = heads * (value_dim + value_dim_edge)
        self.ff_dropout = nn.Dropout(r_dropout)
        #self.ff_norm = nn.LayerNorm(d_attn)
        self.ff_norm = ExChannelNorm(d_attn, affine=True)
        self.ff = FeedForward(d_attn, mult=1, num_layers=nlayer_node)
        self.to_out = nn.Linear(d_attn, d_node)
    
    def get_target_and_source_nodes(self, node, adjmat):
        b, l, d = node.shape
        node_expand = node.expand(b, l, l, d)
        nodetrg = node_expand[adjmat, :].reshape(b, l, -1, d)
        nodesrc = node.unsqueeze(2).expand(b, l, self.nneighbor, d)
        return nodetrg, nodesrc

    def forward(self, x, edgevec, adjmat):
        b, l, _ = x.shape
        ## edge update ##
        nodetrg, nodesrc = self.get_target_and_source_nodes(x, adjmat)
        # concat node-vec & edge-vec
        if(self.nlayer_edge > 0):
            node_edge_node = torch.cat((nodesrc, edgevec, nodetrg), -1)
            edgevec = edgevec + self.edgeupdate(node_edge_node)
        ## node update ##
        x = self.attn_norm(x)
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        # split out heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))
        v_ex = repeat(v, 'b l2 d -> b l1 l2 d', l1=l)
        v = v_ex[repeat(adjmat, '1 l1 l2 -> h l1 l2', h=self.heads), :].reshape(self.heads, l, self.nneighbor, -1)
        # rotary embedding
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        # derive attn logits for scalar and pairwise
        attn_logits_node = torch.einsum('b i d, b j d -> b i j', q, k) * self.attn_logits_scale
        attn_logits_node = attn_logits_node[repeat(adjmat, 'b l1 l2 -> (b h) l1 l2', h=self.heads)].reshape((b*self.heads),-1,self.nneighbor)
        attn_logits_edge = self.to_edge_attn_bias(edgevec) * self.edge_attn_logits_scale
        v_edge = rearrange(self.to_edge_v(edgevec), 'b i j (h d) -> b h i j d', h=self.heads)
        # combine attn logits
        attn_logits = attn_logits_node + attn_logits_edge
        # attention
        attn = attn_logits.softmax(dim = - 1)
        results_node = torch.einsum('b i j, b i j d -> b i d', attn, v)
        attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h = self.heads)
        results_edge = torch.einsum('b h i j, b h i j d -> b h i d', attn_with_heads, v_edge)
        # merge back heads
        results_node = rearrange(results_node, '(b h) n d -> b n (h d)', h = self.heads)
        results_edge = rearrange(results_edge, 'b h n d -> b n (h d)', h = self.heads)
        # concat results and project out
        results = torch.cat([results_node, results_edge], dim = -1)
        results = self.ff(self.ff_norm(self.ff_dropout(results)))
        x = x + self.to_out(results)
        return x, edgevec


