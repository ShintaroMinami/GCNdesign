#! /usr/bin/env python

import sys
from os import path
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

dir_script = path.dirname(path.realpath(__file__))
sys.path.append(dir_script+'/../')
from gcndesign.hypara import HyperParam, InputSource
from gcndesign.dataset import BBGDataset
from gcndesign.training import test
from gcndesign.models import GCNdesign

hypara = HyperParam()
source = InputSource()

# default processing device
source.device = "cuda" if torch.cuda.is_available() else "cpu"

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--list-in', '-l', type=str, default=source.file_list, metavar='[File]',
                    help='Input list file.', required=True)
parser.add_argument('--output', type=str, default=source.file_out, metavar='[File]',
                    help='Output file. (default:"'+source.file_out+'")')
parser.add_argument('--device', type=str, default=source.device, choices=['cpu', 'cuda'],
                    help='Processing device.')
parser.add_argument('--param-in', '-p', type=str, default=source.param_in, metavar='[File]',
                    help='Pre-trained parameter file. (default:{})'.format(source.param_in))

# arguments
args = parser.parse_args()
source.file_list = args.list_in
source.file_out = args.output
source.device = args.device
source.param_in = args.param_in

## Model Setup ##
model = GCNdesign(hypara).to(source.device)
# Network size
params = model.size()

## Test ##
# check fles
assert path.isfile(source.file_list), "List file {:s} was not found.".format(source.file_list)
assert path.isfile(source.param_in), "Parameter file {:s} was not found.".format(source.param_in)

# load pre-trained parameters
model.load_state_dict(torch.load(source.param_in, map_location=torch.device(source.device)), strict=True)

# dataloader setup
test_dataset = BBGDataset(listfile=source.file_list, hypara=hypara)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

# loss function
criterion = nn.CrossEntropyLoss().to(source.device)

# test
loss_test, acc_test = float('inf'), 0
loss_test, acc_test = test(model, criterion, source, test_loader)
print("# Total: Loss: %5.3f  Acc: %6.2f %%" % (loss_test, acc_test))
