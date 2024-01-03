#! /usr/bin/env python

import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
from pathlib import Path


dir_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(dir_path)
from gcndesign.inference import SequenceGenerator

# default hypara
default_checkpoint = f"{dir_path}/gcndesign/data/sakuma20231229-099.ckp"

# default processing device
default_device = "cuda" if torch.cuda.is_available() else "cpu"
device = default_device

# argument parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('pdb', type=str,
                    help='Input PDB file.')
parser.add_argument('--num-sequences', '-n', type=int, default=20,
                    help='Number of sequences to be designed.')
parser.add_argument('--temperature', '-t', type=float, default=0.1,
                    help='Sampling temperature.')
parser.add_argument('--num-res-onestep', type=int, default=10,
                    help='Number of residues generated in one step.')
parser.add_argument('--checkpoint', type=str, default=default_checkpoint,
                    help='Checkpoint file.')
parser.add_argument('--device', type=str, default=default_device, choices=['cpu', 'cuda'],
                    help='Processing device.')
args = parser.parse_args()


# Model
gcn = SequenceGenerator(args.checkpoint)

# Generate
fasta = gcn.design_sequences(
    args.pdb,
    num_sequences=args.num_sequences,
    temperature=args.temperature,
    num_replace=args.num_res_onestep
    )

for i, seq in enumerate(fasta):
    print(f">{i:03d}")
    print(f"{seq}")