import sys
from os import path
import torch
from torch.utils.data import Dataset
import numpy as np
from .pdbutil import ProteinBackbone as pdb
from .hypara import HyperParam
from tqdm import tqdm
import pickle

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
    bb.addO(force=True)
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
    mask = np.ones((len(bb)), dtype=np.bool)
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
    aa1 = np.array([three2one.get(x,'X') for x in res])
    label = np.array([mapped.get(x,20) for x in aa1])
    mask = mask * ~(label == 20)
    # return
    node = torch.FloatTensor(node)
    edgemat = torch.FloatTensor(edgemat)
    adjmat = torch.BoolTensor(adjmat)
    mask = torch.BoolTensor(mask)
    label = torch.LongTensor(label)
    return node, edgemat, adjmat, label, mask, aa1


##  Preprocessing
def Preprocessing(file_list: str, dir_out: str='./', hypara=HyperParam()):
    pdbs = open(file_list, 'r').read().splitlines()
    count = 0
    for pdb in tqdm(pdbs):
        id = path.splitext(path.basename(pdb))[0]
        infile = pdb
        outfile = dir_out + '/' + id + '.pkl'
        count = count + 1
        node, edgemat, adjmat, label, mask, aa1 = pdb2input(infile, hypara)
        with open(outfile, 'wb') as f:
            pickle.dump((node, edgemat, adjmat, label, mask, aa1), f)
    print("\nPre-processing was completed.")
    # return
    return


##  Add head, tail (left, right) margins for data 
def add_margin(node, edgemat, adjmat, label, mask, nneighbor):
    # for node
    node = torch.nn.functional.pad(node, (0,0,1,1), 'constant', 0)
    edgemat = torch.nn.functional.pad(edgemat, (0,0,1,1,1,1), 'constant', 0)
    adjmat = torch.nn.functional.pad(adjmat, (0,0,1,1,1,1), 'constant', False)
    adjmat[0,0:nneighbor,0] = True
    adjmat[-1,0:nneighbor,0] = True
    label = torch.nn.functional.pad(label, (1,1), 'constant', 20)
    mask = torch.nn.functional.pad(mask, (1,1), 'constant', False)
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
        with open(infile, 'rb') as f:
            node, edgemat, adjmat, label, mask, _ = pickle.load(f)
        # add margin
        node, edgemat, adjmat, label, mask = add_margin(node, edgemat, adjmat, label, mask, self.nneighbor)
        # to Torch Tensor
        node = node.squeeze()
        edgemat = edgemat.squeeze()
        adjmat = adjmat.squeeze()
        label = label.squeeze()
        mask = mask.squeeze()
        # return
        return node, edgemat, adjmat, label, mask, self.list_samples[idx]
