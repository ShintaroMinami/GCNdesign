import sys
from os import path
import torch
from torch.utils.data import Dataset
import numpy as np
from pandas import Series as series
from .pdbutil import ProteinBackbone as pdb
from .hypara import HyperParam
from tqdm import tqdm

# Int code of amino-acid types
mapped = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
          'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
          'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
          'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
# 3 letter code to 1 letter code
three2one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU':'E', 'PHE': 'F',
             'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
             'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
             'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

##  PDB data
def pdb2input(filename, hypara):
    bb = pdb(file=filename)
    # add atoms
    bb.addCB(force=True)
    bb.addH(force=True)
    bb.addO()
    bb.coord[0, 5] = bb.coord[0, 0]
    bb.coord[-1, 4] = bb.coord[-1, 3]
    # node features
    node = np.zeros((len(bb), 6), dtype=np.float)
    bb.calc_dihedral()
    sins = np.sin( np.deg2rad(bb.dihedral) )
    coss = np.cos( np.deg2rad(bb.dihedral) )
    node[:, 0::2] = sins
    node[:, 1::2] = coss
    node[0, 0:2] = 0
    node[-1, 2:] = 0
    # mask
    mask = np.ones((len(bb), 1), dtype=np.bool)
    for iaa in range(len(bb)):
        d1 = np.sqrt(np.sum((bb[iaa,0,:] - bb[iaa,1,:])**2, axis=0))
        d2 = np.sqrt(np.sum((bb[iaa,1,:] - bb[iaa,2,:])**2, axis=0))
        if d1 > hypara.dist_chbreak or d2 > hypara.dist_chbreak: mask[iaa] = 0
    for iaa in range(len(bb)-1):
        d3 = np.sqrt(np.sum((bb[iaa,2,:] - bb[iaa+1,0,:])**2, axis=0))
        if d3 > hypara.dist_chbreak: mask[iaa], mask[iaa+1] = 0, 0
    # edge features
    edgemat = np.zeros((len(bb), len(bb), 36), dtype=np.float)
    adjmat = np.zeros((len(bb), len(bb), 1), dtype=np.bool)
    nn = bb.get_nearestN(hypara.nneighbor, atomtype='CB')
    for iaa in range(len(bb)):
        adjmat[iaa, nn[iaa]] = True
        #####
        if(mask[iaa] == False): continue
        #####
        for i in nn[iaa]:
            edgemat[iaa, i] = np.sqrt(
                np.sum((bb[iaa,:,np.newaxis,:] - bb[i,np.newaxis,:,:])**2, axis=2)
            ).reshape(-1)
            edgemat[iaa, i] = (edgemat[iaa, i] - hypara.dist_mean) / hypara.dist_var
    # label
    res = bb.resname
    aa1 = series(res).map(lambda x: three2one.get(x,'X'))
    label = np.array(aa1.map(lambda x: mapped.get(x,-1)), dtype=np.int)
    label = label.reshape(label.shape[0], 1)
    mask = mask * ~(label == -1)
    # return
    return node, edgemat, adjmat, label, mask, aa1


##  Preprocessing
def Preprocessing(file_list: str, dir_out: str='./', hypara=HyperParam()):
    pdbs = open(file_list, 'r').read().splitlines()
    count = 0
    for pdb in pdbs:
        id = path.splitext(path.basename(pdb))[0]
        infile = pdb
        outfile = dir_out + '/' + id + '.csv'
        count = count + 1
        sys.stderr.write('\r\033[K' + '[{}/{}] processing... ({})'.format(count, len(pdbs), infile))
        sys.stderr.flush()
        node, edgemat, adjmat, label, mask, aa1 = pdb2input(infile, hypara)
        with open(outfile, 'w') as f:
            for iaa in range(len(node)):
                feature = ','.join(map(str, np.round(node[iaa], decimals=5)))
                f.write("NODE,%d,%s,%s,%d,%d\n" % (iaa, feature, aa1[iaa], label[iaa], mask[iaa]))
            id1, id2 = np.where(adjmat[:,:,0]==True)
            for i in range(len(id1)):
                feature = ','.join(map(str, np.round(edgemat[id1[i],id2[i]], decimals=5)))
                f.write("EDGE,%d,%d,%s\n" % (id1[i], id2[i], feature))
    print("\nPre-processing was completed.")
    # return
    return


##  Add head, tail (left, right) margins for data 
def add_margin(node, edgemat, adjmat, label, mask, nneighbor):
    # for node
    head_margin = np.zeros((1, node.shape[1]), dtype=np.float)
    tail_margin = np.zeros((1, node.shape[1]), dtype=np.float)
    node = np.concatenate((head_margin, node, tail_margin), axis=0)
    # for label
    head_margin = np.zeros((1, label.shape[1]), dtype=np.int)
    tail_margin = np.zeros((1, label.shape[1]), dtype=np.int)
    label = np.concatenate((head_margin, label, tail_margin), axis=0)
    # for mask
    head_margin = np.zeros((1, mask.shape[1]), dtype=np.bool)
    tail_margin = np.zeros((1, mask.shape[1]), dtype=np.bool)
    mask = np.concatenate((head_margin, mask, tail_margin), axis=0)
    # for edgemat
    head_margin = np.zeros((1, edgemat.shape[1], edgemat.shape[2]), dtype=np.float)
    tail_margin = np.zeros((1, edgemat.shape[1], edgemat.shape[2]), dtype=np.float)
    edgemat = np.concatenate((head_margin, edgemat, tail_margin), axis=0)
    left_margin = np.zeros((edgemat.shape[0], 1, edgemat.shape[2]), dtype=np.float)
    right_margin = np.zeros((edgemat.shape[0], 1, edgemat.shape[2]), dtype=np.float)
    edgemat = np.concatenate((left_margin, edgemat, right_margin), axis=1)
    # for adjmat
    head_margin = np.zeros((1, adjmat.shape[1], adjmat.shape[2]), dtype=np.bool)
    tail_margin = np.zeros((1, adjmat.shape[1], adjmat.shape[2]), dtype=np.bool)
    head_margin[0, 0:nneighbor, 0] = [True]*nneighbor
    tail_margin[0, 0:nneighbor, 0] = [True]*nneighbor
    adjmat = np.concatenate((head_margin, adjmat, tail_margin), axis=0)
    left_margin = np.zeros((adjmat.shape[0], 1, adjmat.shape[2]), dtype=np.bool)
    right_margin = np.zeros((adjmat.shape[0], 1, adjmat.shape[2]), dtype=np.bool)
    adjmat = np.concatenate((left_margin, adjmat, right_margin), axis=1)
    # return
    return node, edgemat, adjmat, label, mask


##  Dataset
class BBGDataset(Dataset):
    def __init__(self, listfile, hypara):
        with open(listfile, 'r') as f:
            self.list_samples = f.read().splitlines()
        self.nneighbor = hypara.nneighbor
    def __len__(self):
        return len(self.list_samples)
    def __getitem__(self, idx):
        infile = self.list_samples[idx]
        with open(infile, 'r') as f:
            lines = f.read().splitlines()
        nodelines = np.array([l.split(',') for l in lines if 'NODE' in l])
        edgelines = np.array([l.split(',') for l in lines if 'EDGE' in l])
        # node info
        _, node, aa1, label, mask = np.hsplit(nodelines, [2, 8, 9, 10])
        node = np.array(node, dtype='float')
        size = len(node)
        label = np.array(label, dtype='int')
        mask = np.array(mask, dtype='int')
        # edge info
        _, row, col, val = np.hsplit(edgelines, [1, 2, 3])
        edgemat = np.zeros((size, size, 36), dtype=np.float)
        adjmat = np.zeros((size, size, 1), dtype=np.bool)
        for i in range(len(row)):
            edgemat[int(row[i])][int(col[i])] = val[i]
            adjmat[int(row[i])][int(col[i])] = 1
        # add margin
        node, edgemat, adjmat, label, mask = add_margin(node, edgemat, adjmat, label, mask, self.nneighbor)
        # to Torch Tensor
        node = torch.FloatTensor(node).squeeze()
        edgemat = torch.FloatTensor(edgemat).squeeze()
        adjmat = torch.BoolTensor(adjmat).squeeze()
        label = torch.LongTensor(label).squeeze()
        mask = torch.BoolTensor(mask).squeeze()
        # return
        return node, edgemat, adjmat, label, mask, self.list_samples[idx]


##  Dataset
class BBGDataset_fast(Dataset):
    def __init__(self, listfile, hypara):
        with open(listfile, 'r') as f:
            self.list_samples = f.read().splitlines()
        self.nneighbor = hypara.nneighbor
        self.data = []
        for sample in tqdm(self.list_samples):
            with open(sample, 'r') as f:
                lines = f.read().splitlines()
            nodelines = np.array([l.split(',') for l in lines if 'NODE' in l])
            edgelines = np.array([l.split(',') for l in lines if 'EDGE' in l])
            # node info
            _, node, aa1, label, mask = np.hsplit(nodelines, [2, 8, 9, 10])
            node = np.array(node, dtype='float')
            size = len(node)
            label = np.array(label, dtype='int')
            mask = np.array(mask, dtype='int')
            # edge info
            _, row, col, val = np.hsplit(edgelines, [1, 2, 3])
            edgemat = np.zeros((size, size, 36), dtype=np.float)
            adjmat = np.zeros((size, size, 1), dtype=np.bool)
            for i in range(len(row)):
                edgemat[int(row[i])][int(col[i])] = val[i]
                adjmat[int(row[i])][int(col[i])] = 1
            # add margin
            node, edgemat, adjmat, label, mask = add_margin(node, edgemat, adjmat, label, mask, self.nneighbor)
            # to Torch Tensor
            node = torch.FloatTensor(node).squeeze()
            edgemat = torch.FloatTensor(edgemat).squeeze()
            adjmat = torch.BoolTensor(adjmat).squeeze()
            label = torch.LongTensor(label).squeeze()
            mask = torch.BoolTensor(mask).squeeze()
            self.data.append((node, edgemat, adjmat, label, mask, sample))
        return
    def __len__(self):
        return len(self.list_samples)
    def __getitem__(self, idx):
        return self.data[idx]
