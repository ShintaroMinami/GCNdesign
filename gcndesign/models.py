import torch
import torch.nn as nn
from .dataset import pdb2input
from .gcn_modules import ExChannelNorm, ResBlock, RGCBlock, RGCBlock_simple, GTFBlock
from einops.layers.torch import Rearrange


##  Main module

##  Main module
class GCNdesign(nn.Module):
    def __init__(self,
        d_node_in: int=6,
        d_edge_in: int=36,
        d_node_out: int=20,
        knn: int=20,
        r_dropout: float=0.2,
        kernel_size: int=9,
        d_node_feat: int=20,
        d_hidden_node_feat: int=64,
        nlayer_node_feat: int=2,
        niter_gcn_encoder: int=4,
        k_node_gcn_encoder: int=20,
        k_edge_gcn_encoder: int=20,
        d_hidden_node_encoder: int=128,
        d_hidden_edge_encoder: int=128,
        nlayer_node_encoder: int=2,
        nlayer_edge_encoder: int=2,
        d_enc_idx_decoder: int=32,
        niter_gcn_decoder: int=4,
        k_node_gcn_decoder: int=20,
        k_edge_gcn_decoder: int=20,
        d_hidden_node_decoder: int=128,
        d_hidden_edge_decoder: int=128,
        nlayer_node_decoder: int=2,
        nlayer_edge_decoder: int=2
        ):
        super().__init__()
        self.d_node_in = d_node_in
        self.d_edge_in = d_edge_in
        self.knn = knn
        self.mask_index = d_node_out
        assert (kernel_size-1)%2 == 0
        padding = int((kernel_size-1)/2)
        ##  node featurize module
        self.nodefeature0 = nn.Sequential(
            Rearrange('b l c -> b c l'),
            nn.Conv1d(d_node_in, d_hidden_node_feat, kernel_size=kernel_size, stride=1, padding=padding),
            Rearrange('b c l -> b l c'),
            *[ResBlock(d_hidden_node_feat, d_hidden_node_feat, dropout=r_dropout) for _ in range(nlayer_node_feat-1)],
            ResBlock(d_hidden_node_feat, d_node_feat, dropout=r_dropout),
            ExChannelNorm(d_node_feat, affine=True),
            nn.ReLU(),
        )
        ##  encoder
        d_node_in_encoder = d_node_feat
        d_edge_in_encoder = d_edge_in
        self.encoder = nn.ModuleList()
        for i in range(niter_gcn_encoder):
            dim_node_in = d_node_in_encoder+k_node_gcn_encoder*i
            dim_node_out = d_node_in_encoder+k_node_gcn_encoder*(i+1)
            dim_edge_in = d_edge_in_encoder+k_edge_gcn_encoder*i
            dim_edge_out = d_edge_in_encoder+k_edge_gcn_encoder*(i+1)
            self.encoder.append(
                RGCBlock(
                    d_node_in=dim_node_in, d_node_out=dim_node_out,
                    d_edge_in=dim_edge_in, d_edge_out=dim_edge_out,
                    d_node_hidden=d_hidden_node_encoder, d_edge_hidden=d_hidden_edge_encoder,
                    nlayer_node=nlayer_node_encoder, nlayer_edge=nlayer_edge_encoder,
                    knn=knn, r_dropout=r_dropout
                )
            )
        ##  decoder
        self.index_encoder = nn.Sequential(
            nn.Embedding(self.mask_index+1, d_enc_idx_decoder),
            ResBlock(d_enc_idx_decoder, d_enc_idx_decoder, dropout=r_dropout),
            ExChannelNorm(d_enc_idx_decoder, affine=True),
            nn.ReLU(),
        )
        d_node_in_decoder = d_node_feat + niter_gcn_decoder * k_node_gcn_decoder + d_enc_idx_decoder
        d_edge_in_decoder = d_edge_in + niter_gcn_decoder * k_edge_gcn_decoder
        self.decoder = nn.ModuleList()
        for i in range(niter_gcn_decoder):
            dim_node_in = d_node_in_decoder+k_node_gcn_decoder*i
            dim_node_out = d_node_in_decoder+k_node_gcn_decoder*(i+1)
            dim_edge_in = d_edge_in_decoder+k_edge_gcn_decoder*i
            dim_edge_out = d_edge_in_decoder+k_edge_gcn_decoder*(i+1)
            self.decoder.append(
                RGCBlock(
                    d_node_in=dim_node_in, d_node_out=dim_node_out,
                    d_edge_in=dim_edge_in, d_edge_out=dim_edge_out,
                    d_node_hidden=d_hidden_node_decoder, d_edge_hidden=d_hidden_edge_decoder,
                    nlayer_node=nlayer_node_decoder, nlayer_edge=nlayer_edge_decoder,
                    knn=knn, r_dropout=r_dropout
                )
            )
        ##  to output
        d_node_out_decoder = d_node_in_decoder + niter_gcn_decoder * k_node_gcn_decoder
        self.to_out = nn.Linear(d_node_out_decoder, d_node_out)

    def encode(self, node, edgemat, adjmat):
        b, l, _ = node.shape
        # node feature
        node = self.nodefeature0(node)
        edge = edgemat[adjmat, :].reshape(b,l,-1,self.d_edge_in)
        # encoder
        for f in self.encoder:
            node, edge = f(node, edge, adjmat)
        return node, edge

    def decode(self, node, edge, adjmat, masked_resid=None):
        if masked_resid == None:
            masked_resid = torch.ones(node.shape[:2], device=node.device, dtype=int) * self.mask_index
        feat_idx_enc = self.index_encoder(masked_resid)
        node = torch.cat([node, feat_idx_enc], dim=-1)
        for f in self.decoder:
            node, edge = f(node, edge, adjmat)
        return node, edge

    def forward(self, node_in, edgemat_in, adjmat_in, masked_resid):
        # encoder
        node, edge = self.encode(node_in, edgemat_in, adjmat_in)
        # decoder
        node, _ = self.decode(node, edge, adjmat_in, masked_resid)
        # to output        
        out = self.to_out(node)
        return out

    def process_pdbfile(self, pdbfile, require_all=False):
        node, edgemat, adjmat, label, mask, res = pdb2input(pdbfile, knn=self.knn)
        if require_all:
            return node, edgemat, adjmat, label, mask, res
        else:
            return node, edgemat, adjmat

    def size(self):
        params = 0
        for p in self.parameters():
            if p.requires_grad:
                params += p.numel()
        return params



class GCNdesign_simple(nn.Module):
    def __init__(self,
        d_node_in: int=6,
        d_edge_in: int=36,
        d_node_out: int=20,
        knn: int=20,
        r_dropout: float=0.2,
        kernel_size: int=9,
        d_node_encode: int=128,
        d_edge_encode: int=32,
        nlayer_node_feat: int=2,
        nlayer_edge_feat: int=2,
        niter_gcn_encoder: int=4,
        nlayer_node_encoder: int=2,
        nlayer_edge_encoder: int=2,
        d_enc_idx_decoder: int=32,
        niter_gcn_decoder: int=4,
        nlayer_node_decoder: int=2,
        nlayer_edge_decoder: int=2
        ):
        super().__init__()
        self.d_node_in = d_node_in
        self.d_edge_in = d_edge_in
        self.knn = knn
        self.mask_index = d_node_out
        assert (kernel_size-1)%2 == 0
        padding = int((kernel_size-1)/2)
        ##  node featurize module
        self.nodefeature0 = nn.Sequential(
            Rearrange('b l c -> b c l'),
            nn.Conv1d(d_node_in, d_node_encode, kernel_size=kernel_size, stride=1, padding=padding),
            Rearrange('b c l -> b l c'),
            *[ResBlock(d_node_encode, d_node_encode, dropout=r_dropout) for _ in range(nlayer_node_feat-1)],
            ResBlock(d_node_encode, d_node_encode, dropout=r_dropout),
            ExChannelNorm(d_node_encode, affine=True),
            nn.ReLU(),
        )
        self.edgefeature0 = nn.Sequential(
            *[ResBlock(d_edge_in, d_edge_encode, dropout=r_dropout) for _ in range(nlayer_edge_feat-1)],
            ResBlock(d_edge_encode, d_edge_encode, dropout=r_dropout),
            ExChannelNorm(d_edge_encode, affine=True),
            nn.ReLU(),
        )
        ##  encoder
        self.encoder = nn.ModuleList()
        for _ in range(niter_gcn_encoder):
            self.encoder.append(
                RGCBlock_simple(
                    d_node=d_node_encode, d_edge=d_edge_encode,
                    nlayer_node=nlayer_node_encoder, nlayer_edge=nlayer_edge_encoder,
                    knn=knn, r_dropout=r_dropout
                )
            )
        ##  decoder
        self.index_encoder = nn.Sequential(
            nn.Embedding(self.mask_index+1, d_enc_idx_decoder),
            ResBlock(d_enc_idx_decoder, d_enc_idx_decoder, dropout=r_dropout),
            ExChannelNorm(d_enc_idx_decoder, affine=True),
            nn.ReLU(),
        )
        d_node_decode, d_edge_decode = d_node_encode + d_enc_idx_decoder, d_edge_encode
        self.decoder = nn.ModuleList()
        for i in range(niter_gcn_decoder):
            self.decoder.append(
                RGCBlock_simple(
                    d_node=d_node_decode, d_edge=d_edge_decode,
                    nlayer_node=nlayer_node_decoder, nlayer_edge=nlayer_edge_decoder,
                    knn=knn, r_dropout=r_dropout
                )
            )
        ##  to output
        self.to_out = nn.Linear(d_node_decode, d_node_out)

    def encode(self, node, edgemat, adjmat):
        b, l, _ = node.shape
        # node feature
        node = self.nodefeature0(node)
        edge = edgemat[adjmat, :].reshape(b,l,-1,self.d_edge_in)
        edge = self.edgefeature0(edge)
        # encoder
        for f in self.encoder:
            node, edge = f(node, edge, adjmat)
        return node, edge

    def decode(self, node, edge, adjmat, masked_resid=None):
        if masked_resid == None:
            masked_resid = torch.ones(node.shape[:2], device=node.device, dtype=int) * self.mask_index
        feat_idx_enc = self.index_encoder(masked_resid)
        node = torch.cat([node, feat_idx_enc], dim=-1)
        for f in self.decoder:
            node, edge = f(node, edge, adjmat)
        return node, edge

    def forward(self, node_in, edgemat_in, adjmat_in, masked_resid):
        # encoder
        node, edge = self.encode(node_in, edgemat_in, adjmat_in)
        # decoder
        node, _ = self.decode(node, edge, adjmat_in, masked_resid)
        # to output        
        out = self.to_out(node)
        return out

    def process_pdbfile(self, pdbfile, require_all=False):
        node, edgemat, adjmat, label, mask, res = pdb2input(pdbfile, knn=self.knn)
        if require_all:
            return node, edgemat, adjmat, label, mask, res
        else:
            return node, edgemat, adjmat

    def size(self):
        params = 0
        for p in self.parameters():
            if p.requires_grad:
                params += p.numel()
        return params

