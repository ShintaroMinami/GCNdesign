import torch
import torch.nn as nn
from .dataset import pdb2input

##  Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('InstanceNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        m.bias.data.fill_(0)


##  ResBlock with InstanceNormalization
class ResBlock_InstanceNorm(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.2):
        super(ResBlock_InstanceNorm, self).__init__()
        #  layer1
        self.bn1 = nn.InstanceNorm1d(d_in, affine=True)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(d_in, d_out, kernel_size=1, stride=1, padding=0)
        #  layer2
        self.bn2 = nn.InstanceNorm1d(d_out, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(d_out, d_out, kernel_size=1, stride=1, padding=0)
        #  shortcut
        self.shortcut = nn.Sequential()
        if d_in != d_out:
            self.shortcut.add_module('bn', nn.InstanceNorm1d(d_in, affine=True))
            self.shortcut.add_module('conv', nn.Conv1d(d_in, d_out, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.dropout2(self.relu2(self.bn2(out))))
        out += self.shortcut(x)
        return out


##  ResBlock with BatchNormalization
class ResBlock_BatchNorm(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.2):
        super(ResBlock_BatchNorm, self).__init__()
        #  layer1
        self.bn1 = nn.BatchNorm1d(d_in, affine=True)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv1d(d_in, d_out, kernel_size=1, stride=1, padding=0)
        #  layer2
        self.bn2 = nn.BatchNorm1d(d_out, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(d_out, d_out, kernel_size=1, stride=1, padding=0)
        #  shortcut
        self.shortcut = nn.Sequential()
        if d_in != d_out:
            self.shortcut.add_module('bn', nn.BatchNorm1d(d_in, affine=True))
            self.shortcut.add_module('conv', nn.Conv1d(d_in, d_out, kernel_size=1, stride=1, padding=0))
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.dropout2(self.relu2(self.bn2(out))))
        out += self.shortcut(x)
        return out


## Graph Convolution Block (DenseNet-like update)
class RGCBlock(nn.Module):
    def __init__(self, d_in, d_out, d_edge_in, d_edge_out, nneighbor,
                 d_hidden_node, d_hidden_edge, nlayer_node, nlayer_edge, dropout):
        super(RGCBlock, self).__init__()
        self.nlayer_edge = nlayer_edge
        self.d_in = d_in
        self.d_out = d_out
        self.k_node = d_out - d_in
        self.k_edge = d_edge_out - d_edge_in
        self.nneighbor = nneighbor
        self.d_hidden_att = 2
        #  edge update layer
        if(nlayer_edge > 0):
            self.edgeupdate = nn.ModuleList(
                [nn.Conv1d(d_edge_in+d_in+d_in, d_hidden_edge, kernel_size=1, stride=1, padding=0)] +
                [ResBlock_BatchNorm(d_hidden_edge, d_hidden_edge, dropout=dropout) for _ in range(nlayer_edge)] +
                [ResBlock_BatchNorm(d_hidden_edge, self.k_edge, dropout=dropout)] +
                [nn.BatchNorm1d(self.k_edge, affine=True)] +
                [nn.ReLU()]
            )
        #  graph convolution layer
        self.encoding = nn.ModuleList(
            [nn.Conv1d(d_edge_out+2*d_in, d_hidden_node, kernel_size=1, stride=1, padding=0)] +
            [ResBlock_BatchNorm(d_hidden_node, d_hidden_node, dropout=dropout) for _ in range(nlayer_node)] +
            [nn.BatchNorm1d(d_hidden_node, affine=True)] +
            [nn.ReLU()]
        )
        #  residual update layer
        self.residual = nn.ModuleList(
            [ResBlock_BatchNorm(d_hidden_node, d_hidden_node, dropout=dropout) for _ in range(nlayer_node)] +
            [ResBlock_BatchNorm(d_hidden_node, self.k_node, dropout=dropout)] +
            [nn.BatchNorm1d(self.k_node, affine=True)] +
            [nn.ReLU()]
        )
            
    def forward(self, x, edgevec, adjmat):
        naa = adjmat.size()[0]
        # node-vec
        node_expand = x.unsqueeze(0).expand(naa, naa, self.d_in)
        nodetrg = node_expand[adjmat, :].reshape(naa, -1, self.d_in)
        nodesrc = x.unsqueeze(1).expand(naa, self.nneighbor, self.d_in)
        ## edge update ##
        # concat node-vec & edge-vec
        if(self.nlayer_edge > 0):
            selfnode = x.unsqueeze(1).expand(naa, self.nneighbor, self.d_in)
            nen = torch.cat((selfnode, edgevec, nodetrg), 2).transpose(1, 2)
            for f in self.edgeupdate:
                nen = f(nen)
            edgevec_new = nen.transpose(1, 2)
            edgevec = torch.cat((edgevec, edgevec_new), 2)
        ## node update ##
        nodeedge = torch.cat((nodesrc, edgevec, nodetrg), 2).transpose(1, 2)
        # encoding layer
        encoded = nodeedge
        for f in self.encoding:
            encoded = f(encoded)
        aggregated = encoded.sum(2)
        residual = aggregated.unsqueeze(2)
        for f in self.residual:
            residual = f(residual)
        residual = residual.squeeze(2)
        # add dense connection
        out = torch.cat((x, residual), 1)
        # return
        return out, edgevec


##  Embedding module
class Embedding_module(nn.Module):
    def __init__(self, nneighbor, r_drop,
                 d_node0, d_hidden_node0, nlayer_node0,
                 d_hidden_node, d_hidden_edge, nlayer_node, nlayer_edge,
                 niter_rgc, k_node_rgc, k_edge_rgc, fragment_size):
        super(Embedding_module, self).__init__()
        self.d_node_in = 6  # fix (sin(phi), cos(phi), sin(psi), cos(psi), sin(ome), cos(ome))
        self.d_edge_in = 36 # fix (matrix; 6 atoms x 6 atoms)
        assert (fragment_size-1)%4 == 0
        kernel_size = int((fragment_size-1)/2 + 1)
        padding = int((kernel_size-1)/2)
        if(nlayer_edge < 1):
            k_edge_rgc = 0
        
        self.nodefeature0 = nn.ModuleList(
            [nn.Conv1d(self.d_node_in, d_hidden_node0, kernel_size=kernel_size, stride=1, padding=padding)] +
            [ResBlock_InstanceNorm(d_hidden_node0, d_hidden_node0, dropout=r_drop) for _ in range(nlayer_node0)] +
            [ResBlock_InstanceNorm(d_hidden_node0, d_node0, dropout=r_drop)] +
            [nn.InstanceNorm1d(d_node0, affine=True)] +
            [nn.ReLU()]
        )
        self.rgclayer = nn.ModuleList(
            [RGCBlock(d_node0+k_node_rgc*i, d_node0+k_node_rgc*(i+1), self.d_edge_in+k_edge_rgc*i, self.d_edge_in+k_edge_rgc*(i+1),
                      nneighbor, d_hidden_node, d_hidden_edge, nlayer_node, nlayer_edge, r_drop) for i in range(niter_rgc)]
        )

    def forward(self, node_in, edgemat_in, adjmat_in):
        naa = node_in.size()[0]
        # edge
        edge = edgemat_in[adjmat_in, :].reshape(naa, -1, self.d_edge_in)
        # node embedding
        node = node_in.transpose(0, 1).unsqueeze(0)
        for f in self.nodefeature0:
            node = f(node)
        node = node.squeeze(0).transpose(0, 1)
        # Graph Convolution
        for f in self.rgclayer:
            node, edge = f(node, edge, adjmat_in)
        # output
        return node, edge


##  Prediction module (Iterative 1D convolution)
class Prediction_module(nn.Module):
    def __init__(self, d_in, d_out, d_hidden1, d_hidden2, nlayer_pred, fragment_size, r_drop):
        super(Prediction_module, self).__init__()
        assert (fragment_size-1)%4 == 0
        stride = 1
        kernel_size = int((fragment_size-1)/2 + 1)
        padding = int((kernel_size-1)/2)
        self.pred1Dconv = nn.ModuleList(
            [nn.Conv1d(d_in, d_hidden1, kernel_size=kernel_size, stride=stride, padding=padding)] +
            [ResBlock_InstanceNorm(d_hidden1, d_hidden1, dropout=r_drop) for _ in range(nlayer_pred)] +
            [nn.InstanceNorm1d(d_hidden1, affine=True)] +
            [nn.ReLU()] +
            [nn.Conv1d(d_hidden1, d_hidden2, kernel_size=kernel_size, stride=stride, padding=padding)] +
            [ResBlock_InstanceNorm(d_hidden2, d_hidden2, dropout=r_drop) for _ in range(nlayer_pred)] +
            [nn.InstanceNorm1d(d_hidden2, affine=True)] +
            [nn.ReLU()] +
            [nn.Conv1d(d_hidden2, d_out, kernel_size=1, stride=1, padding=0)]
        )
    def forward(self, node_in):
        node_out = node_in.transpose(0, 1).unsqueeze(0)
        # prediction layer
        for f in self.pred1Dconv:
            node_out = f(node_out)
        # output
        node_out = node_out.squeeze(0).transpose(0, 1)
        return(node_out)


##  Main module
class GCNdesign(nn.Module):
    def __init__(self, hypara):
        super(GCNdesign, self).__init__()
        self.hypara = hypara
        ##  embedding module  ##
        self.embedding = Embedding_module(nneighbor=hypara.nneighbor,
                                          r_drop=hypara.r_drop,
                                          d_node0=hypara.d_embed_node0,
                                          fragment_size=hypara.fragment_size0,
                                          d_hidden_node0=hypara.d_embed_h_node0,
                                          nlayer_node0=hypara.nlayer_embed_node0,
                                          d_hidden_node=hypara.d_embed_h_node,
                                          d_hidden_edge=hypara.d_embed_h_edge,
                                          nlayer_node=hypara.nlayer_embed_node,
                                          nlayer_edge=hypara.nlayer_embed_edge,
                                          k_node_rgc=hypara.k_node_rgc,
                                          k_edge_rgc=hypara.k_edge_rgc,
                                          niter_rgc=hypara.niter_embed_rgc)
        ##  prediction module  ##
        self.prediction = Prediction_module(d_in=hypara.d_embed_node0 + hypara.k_node_rgc * hypara.niter_embed_rgc,
                                            d_out=hypara.d_pred_out,
                                            r_drop=hypara.r_drop,
                                            d_hidden1=hypara.d_pred_h1,
                                            d_hidden2=hypara.d_pred_h2,
                                            nlayer_pred=hypara.nlayer_pred,
                                            fragment_size=hypara.fragment_size)
    def size(self):
        params = 0
        for p in self.parameters():
            if p.requires_grad:
                params += p.numel()
        return params
        
    def forward(self, node_in, edgemat_in, adjmat_in):
        # embedding
        latent, _ = self.embedding(node_in, edgemat_in, adjmat_in)
        # prediction
        out = self.prediction(latent)
        # output
        return out

    def get_embedding(self, node_in, edgemat_in, adjmat_in):
        return self.embedding(node_in, edgemat_in, adjmat_in)

    def process_pdbfile(self, pdbfile, require_all=False):
        node, edgemat, adjmat, label, mask, res = pdb2input(pdbfile)
        if require_all:
            return node, edgemat, adjmat, label, mask, res
        else:
            return node, edgemat, adjmat