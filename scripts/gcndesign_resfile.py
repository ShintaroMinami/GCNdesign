#! /usr/bin/env python

import sys
from os import path
import argparse
import torch

dir_script = path.dirname(path.realpath(__file__))
sys.path.append(dir_script+'/../')
from gcndesign.predictor import Predictor

# default processing device
device = "cuda" if torch.cuda.is_available() else "cpu"

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('pdb', type=str, default=None, metavar='[File]',
                    help='PDB file input.')
parser.add_argument('--prob-cut', '-c', type=float, default=0.8, metavar='[Float]',
                    help='Probability cutoff. (default:{})'.format(0.8))
parser.add_argument('--device', type=str, default=device, choices=['cpu', 'cuda'],
                    help='Processing device. (default:\'cuda\' if available)')
parser.add_argument('--param-in', '-p', type=str, default=None, metavar='[File]',
                    help='NN parameter file. (default:{})'.format(None))
args = parser.parse_args()

# check files
assert path.isfile(args.pdb), "PDB file {:s} was not found.".format(args.pdb)
    
# predictor
predictor = Predictor(device=args.device, param=args.param_in)

# resfile
print(predictor.make_resfile(pdb=args.pdb, prob_cut=args.prob_cut))