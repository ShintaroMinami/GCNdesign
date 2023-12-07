#! /usr/bin/env python

import sys
from os import path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

dir_script = path.dirname(path.realpath(__file__))
sys.path.append(dir_script+'/../')
from gcndesign.hypara2021 import HyperParam, InputSource
from gcndesign.models import GCNdesign2021
from gcndesign.dataset import BBGDataset
from gcndesign.training import test

hypara = HyperParam()
source = InputSource()

# default processing device
source.device = "cuda" if torch.cuda.is_available() else "cpu"

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('testlist', type=str, metavar='File',
                    help='List of training data.')
parser.add_argument('--device', type=str, default=source.device, choices=['cpu', 'cuda'],
                    help='Processing device (default:\'cuda\' if available).')
parser.add_argument('--param', type=str, default=None,
                    help='Parameter file.')

##  arguments  ##
args = parser.parse_args()

#  check input
assert path.isfile(args.testlist), "Training data file {:s} was not found.".format(args.testlist)

param = source.param_in if args.param is None else args.param
## Model Setup ##
model = GCNdesign2021(hypara)
model.load_state_dict(torch.load(param, map_location=torch.device(args.device)))

# dataloader setup
test_dataset = BBGDataset(listfile=args.testlist, hypara=hypara)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# loss function
criterion = nn.CrossEntropyLoss().to(source.device)

# test
loss, acc = test(model, criterion, source, test_loader)
print(f"--------------------------------------------------------")
print(f"Per-Residue Loss: {loss:.3f}   Per-Residue Accuracy: {acc:.3f} %")