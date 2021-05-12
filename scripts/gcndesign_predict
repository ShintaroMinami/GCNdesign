#! /usr/bin/env python

import sys
from os import path
import argparse
import torch
import torch.nn as nn

dir_script = path.dirname(path.realpath(__file__))
sys.path.append(dir_script+'/../')
from gcndesign.hypara import HyperParam, InputSource
from gcndesign.models import GCNdesign
from gcndesign.dataset import pdb2input, add_margin

# int code to amino-acid types
i2aa = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')

# parameters
hypara = HyperParam()
source = InputSource()

# default processing device
source.device = "cuda" if torch.cuda.is_available() else "cpu"

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('pdb_in', type=str, default=None, metavar='[File]',
                    help='PDB file input.')
parser.add_argument('--output', type=str, default=source.file_out, metavar='[File]',
                    help='Output file. (default:"'+source.file_out+'")')
parser.add_argument('--device', type=str, default=source.device, choices=['cpu', 'cuda'],
                    help='Processing device.')
parser.add_argument('--param-in', '-p', type=str, default=source.param_in, metavar='[File]',
                    help='Pre-trained parameter file. (default:{})'.format(source.param_in))
parser.add_argument('--resfile', type=str, default=source.resfile_out, metavar='[File]',
                    help='Resfile format file for RosettaDesign.')

# arguments
args = parser.parse_args()
source.pdb_in = args.pdb_in
source.file_out = args.output
source.device = args.device
source.param_in = args.param_in
source.resfile_out = args.resfile

## Model Setup ##
model = GCNdesign(hypara).to(source.device)
# Network size
params = model.size()

# check fles
assert path.isfile(source.pdb_in), "PDB file {:s} was not found.".format(source.pdb_in)
assert path.isfile(source.param_in), "Parameter file {:s} was not found.".format(source.param_in)

# load pre-trained parameters
model.load_state_dict(torch.load(source.param_in, map_location=torch.device(source.device)), strict=True)
    
# input data setup
dat1, dat2, dat3, label, mask, aa1 = pdb2input(source.pdb_in, hypara)
dat1, dat2, dat3, label, mask = add_margin(dat1, dat2, dat3, label, mask, hypara.nneighbor)
dat1 = torch.FloatTensor(dat1).squeeze().to(source.device)
dat2 = torch.FloatTensor(dat2).squeeze().to(source.device)
dat3 = torch.BoolTensor(dat3).squeeze().to(source.device)
label = torch.LongTensor(label).squeeze().to(source.device)
mask = torch.BoolTensor(mask).squeeze().to(source.device)

## prediction ##
model.eval()
outputs = model(dat1, dat2, dat3)
prob = torch.softmax(outputs, dim=1)
maxval = torch.argmax(outputs, dim=1)
for i in range(1, len(prob)-1):
    print(' %4d %s %s:pred ' % (i, aa1[i-1], i2aa[maxval[i]]), end='')
    for ia in range(20):
        print(' %5.3f:%s' % (prob[i][ia], i2aa[ia]), end='')
    print('  %d:mask' % (mask[i]), end="\n")
if source.resfile_out is not None:
    with open(source.resfile_out, 'w') as file:
        sorted_args = torch.argsort(-prob)
        for iaa in range(1, len(prob)-1):
            file.write(' {:4d} {:s} PIKAA  '.format(iaa, 'A'))
            pikaa = ''
            psum = 0
            for i in range(20):
                iarg = sorted_args[iaa][i]
                pikaa = pikaa + i2aa[iarg]
                psum += prob[iaa][iarg]
                if i > 0 and psum > source.prob_cut: break
            file.write('{:20s} # {:s}\n'.format(pikaa, aa1[iaa-1]))
