#! /usr/bin/env python

import sys
from os import path
import argparse
dir_script = path.dirname(path.realpath(__file__))
sys.path.append(dir_script+'/../')
from gcndesign.dataset import Preprocessing

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--list-in', '-l', required=True, type=str, default=None, metavar='[File]',
                    help='List of PDB structures.')
parser.add_argument('--dir-out', '-o', type=str, default='./', metavar='[Directory]',
                    help='Directory in which data processed will be stored.')
args = parser.parse_args()

# check
assert path.isfile(args.list_in), "Input file {:s} is not found.".format(args.list_in)

# preprocessing
Preprocessing(args.list_in, args.dir_out)
