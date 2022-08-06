#! /usr/bin/env python

import sys
from os import path
import argparse
import torch

dir_script = path.dirname(path.realpath(__file__))
sys.path.append(dir_script+'/../')
from gcndesign.predictor import Predictor
from gcndesign.resfile import fix_native_resfile, expand_nums

# default processing device
device = "cuda" if torch.cuda.is_available() else "cpu"

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('pdb', type=str, default=None, metavar='[File]',
                    help='PDB file input.')
parser.add_argument('--temperature', '-t', type=float, default=1.0, metavar='[Float]',
                    help='Temperature: probability P(AA) is proportional to exp(logit(AA)/T). (default:{})'.format(1.0))
parser.add_argument('--prob-cut', '-c', type=float, default=0.6, metavar='[Float]',
                    help='Probability cutoff. (default:{})'.format(0.6))
parser.add_argument('--device', type=str, default=device, choices=['cpu', 'cuda'],
                    help='Processing device. (default:\'cuda\' if available)')
parser.add_argument('--keep', '-k', type=str, default=[], metavar='Str', nargs='+',
                    help='Residue numbers & chain id for keeping the initial amino-acid type. '
                         '(e.g. "-k 1A 2A 3B 11C-15C @D ...", @ represents all residues in the chain). '
                         'Note that "-k 1 3-5 @" is interpreted as "-k 1A 3A-5A @A".')
parser.add_argument('--keep-type', '-kt', type=str, default='NATRO', metavar='String', choices=['NATRO', 'NATAA'],
                    help='(default:{})'.format('NATRO'))
parser.add_argument('--unused', '-u', type=str, default=None, metavar='Char', nargs='+',
                    help='Residue types not to be used. (e.g. "-e C H W ...")')
parser.add_argument('--include-init-restype', default=False, action='store_true',
                    help='Include the initial residue type. (default:{})'.format(False))
parser.add_argument('--param-in', '-p', type=str, default=None, metavar='[File]',
                    help='NN parameter file. (default:{})'.format(None))
args = parser.parse_args()

# check files
assert path.isfile(args.pdb), "PDB file {:s} was not found.".format(args.pdb)
    
# predictor
predictor = Predictor(device=args.device, param=args.param_in)
resfile = predictor.make_resfile(pdb=args.pdb, prob_cut=args.prob_cut, unused=args.unused, temperature=args.temperature)
resfile = fix_native_resfile(resfile, resnums=expand_nums(args.keep), keeptype=args.keep_type)

# output
print(resfile)