#! /usr/bin/env python

import sys
from os import path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json

dir_path = str(Path(__file__).resolve().parent.parent)
sys.path.append(dir_path)
from gcndesign.dataset import BBGDataset
from gcndesign.training import train, valid
from gcndesign.models import GCNdesign

# default hypara
default_hypara = f"{dir_path}/gcndesign/data/default_hypara.json"

# default processing device
default_device = "cuda" if torch.cuda.is_available() else "cpu"

# argument parser
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--train_list', '-t', type=str, metavar='[File, File, ...]',
                    help='List of training data.', required=True)
parser.add_argument('--valid_list', '-v', type=str, metavar='[File, File, ...]',
                    help='List of validation data.', required=True)
parser.add_argument('--hypara-json', '-j', type=str, default=default_hypara, metavar='[File]',
                    help='Hyper parameter setting file (.json).')
parser.add_argument('--nepoch', '-e', type=int, default=100, metavar='[Int]',
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', '-lr', type=float, default=0.002, metavar='[Float]',
                    help='Learning rate.')
parser.add_argument('--checkpoint-in', type=str, default=None, metavar='[File]',
                    help='Checkpoint file.')
parser.add_argument('--output-file', '-o', type=str, default='training_curve.dat', metavar='[File]',
                    help='Output file.')
parser.add_argument('--output-params-prefix', '-p', type=str, default='params', metavar='[prefix]',
                    help='Prefix string for parameter/checkpoint files output.')
parser.add_argument('--monitoring-ratios', type=float, default=[0.0, 0.5, 0.95], help='Available residue ratio for validation.')
parser.add_argument('--max-size', type=int, default=1000, metavar='[Int]',
                    help='Max size of protein')
parser.add_argument('--device', type=str, default=default_device, choices=['cpu', 'cuda'],
                    help='Processing device.')
args = parser.parse_args()

#  check input
assert path.isfile(args.train_list), f"Training data file {args.train_list} was not found."
assert path.isfile(args.valid_list), f"Validation data file {args.valid_list} was not found."

# if checkpoint
if args.checkpoint_in != None:
    checkpoint = torch.load(args.checkpoint_in)
    model_hypara = checkpoint['model_hypara']
    model = GCNdesign(**model_hypara)
    params = model.size()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.nepoch-10, gamma=0.1)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch_init = checkpoint['epoch']+1
else:
    ## Model Setup ##
    model_hypara = json.load(open(args.hypara_json, 'r'))['model_hypara']
    model = GCNdesign(**model_hypara).to(args.device)
    # Network size
    params = model.size()
    epoch_init = 1
    # optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.nepoch-10, gamma=0.1)

# dataloader setup
train_dataset = BBGDataset(listfile=args.train_list, knn=model_hypara['knn'])
valid_dataset = BBGDataset(listfile=args.valid_list, knn=model_hypara['knn'])
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True)

# loss function
criterion = nn.CrossEntropyLoss().to(args.device)

# training routine
file = open(args.output_file, 'w')
file.write("# Total Parameters : {:.2f}M\n".format(params/1000000))
print("# Total Parameters : {:.2f}M".format(params/1000000))
file.flush()
for iepoch in range(epoch_init, args.nepoch):
    loss_train, acc_train, loss_valid, acc_valid = float('inf'), 0, float('inf'), 0
    # training
    loss_train, acc_train = train(model, criterion, train_loader, optimizer, maxsize=args.max_size, device=args.device)
    # validation
    loss_valid, acc_valid = valid(model, criterion, valid_loader, device=args.device, check_ratios=args.monitoring_ratios)
    scheduler.step()
    file.write(f' {iepoch:3d}  T.Loss: {loss_train:5.3f}  T.Acc: {acc_train:5.2f} ')
    file.write("  V.Loss:")
    for ratio in args.monitoring_ratios:
        file.write(f" {loss_valid[ratio]:5.3f}")
    file.write("  V.Acc:")
    for ratio in args.monitoring_ratios:
        file.write(f" {acc_valid[ratio]:5.2f}")
    file.write("\n")
    file.flush()
    # output params
    torch.save(model.to('cpu').state_dict(), "{}-{:03d}.pkl".format(args.output_params_prefix, iepoch))
    torch.save({
        'epoch': iepoch,
        'model_state_dict': model.to('cpu').state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss_train,
        'model_hypara': model_hypara
    }, "{}-{:03d}.ckp".format(args.output_params_prefix, iepoch))
    model.to(args.device)