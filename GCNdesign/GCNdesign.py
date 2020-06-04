#! /usr/bin/env python

import sys
import os
import argparse
import dataclasses
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import *
from training import *
from models import *
from radam import *


##  Default Processing Device  ##
device_default = "cuda" if torch.cuda.is_available() else "cpu"


##  Hyper Parameters  ##
@dataclasses.dataclass
class HyperParam:
    nepoch:             int = 50
    learning_rate:    float = 0.002
    batchsize_cut:      int = 1000
    # protein structure #
    dist_chbreak:     float = 2.0 # distance cutoff for chain break
    dist_mean:        float = 6.4 # distance mean for normalization
    dist_var:         float = 2.4 # distance variance for normalization
    # model structure #
    nneighbor:          int =  20 # 20 neighbors
    d_pred_out:         int =  20 # 20 types of amino-acid
    # dropout #
    r_drop:           float = 0.2
    # for 1st embedding layer #
    fragment_size0:     int =   9
    d_embed_node0:      int =  20
    d_embed_h_node0:    int =  40
    nlayer_embed_node0: int =   4
    # for GCN embedding layers #
    niter_embed_rgc:    int =   8
    k_node_rgc:         int =  20
    k_edge_rgc:         int =  10
    d_embed_h_node:     int = 128
    d_embed_h_edge:     int = 128
    nlayer_embed_node:  int =   4
    nlayer_embed_edge:  int =   4
    # for prediction layer #
    fragment_size:      int =   9
    d_pred_h1:          int = 128
    d_pred_h2:          int =  64
    nlayer_pred:        int =   8
# instance
hypara = HyperParam()


##  Input  ##
@dataclasses.dataclass
class InputSource:
    mode:        str = 'prediction'
    pdb_in:      str = None
    file_list:   str = None
    file_train:  str = None
    file_valid:  str = None
    file_out:    str = 'result.dat'
    dir_out:     str = None
    dir_in:      str = None
    param_out:   str = 'params_out.pkl'
    param_in:    str = 'param_default.pkl'
    onlypred:   bool = False
    resfile_out: str = None
    prob_cut:  float = 0.80
    device:      str = device_default
# instance
source = InputSource()


##  Argument Parser  ##
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default=source.mode,
                    choices=['prediction','training', 'test', 'preprocessing'])
parser.add_argument('--epochs', type=int, default=hypara.nepoch,
                    help='Number of training epochs. [default:{}]'.format(hypara.nepoch))
parser.add_argument('--in_pdb', type=str, default=source.pdb_in,
                    help='PDB file input.')
parser.add_argument('--in_list', type=str, default=source.file_list,
                    help='List of data to be processed.')
parser.add_argument('--in_train_list', type=str, default=source.file_train,
                    help='List of training data.')
parser.add_argument('--in_valid_list', type=str, default=source.file_valid,
                    help='List of validation data.')
parser.add_argument('--in_dir', type=str, default=source.dir_in,
                    help='Directory in which data are stored.')
parser.add_argument('--out_dir', type=str, default=source.dir_out,
                    help='Directory in which data processed will be stored.')
parser.add_argument('--out_resfile', type=str, default=source.resfile_out,
                    help='Resfile format file for RosettaDesign.')
parser.add_argument('--in_params', type=str, default=source.param_in,
                    help='Pre-trained parameter file.')
parser.add_argument('--only_predmodule', action='store_true',
                    help=argparse.SUPPRESS)
parser.add_argument('--out_params', type=str, default=source.param_out,
                    help='Trained parameter file. [default:"'+source.param_out+'"]')
parser.add_argument('--output', type=str, default=source.file_out,
                    help='Output file. [default:"'+source.file_out+'"]')
parser.add_argument('--device', type=str, default=source.device,
                    help='Processing device.')
parser.add_argument('--layer', type=int, default=hypara.niter_embed_rgc,
                    help='Number of GCN layers. [default:{}]'.format(hypara.niter_embed_rgc))
##  Arguments  ##
args = parser.parse_args()
source.mode = args.mode
hypara.nepoch = args.epochs
source.pdb_in = args.in_pdb
source.file_list = args.in_list
source.file_train = args.in_train_list
source.file_valid = args.in_valid_list
source.dir_in = args.in_dir
source.dir_out = args.out_dir
source.param_in = args.in_params
source.onlypred = args.only_predmodule
source.param_out = args.out_params
source.resfile_out = args.out_resfile
source.file_out = args.output
source.device = args.device
hypara.niter_embed_rgc = args.layer


##################
###            ###
###    Main    ###
###            ###
##################

###  Preprocessing Mode  ###
if source.mode == 'preprocessing':
    if (source.file_list is None) or (not os.path.isfile(source.file_list)):
        print("Input file [%s] is not found." % (source.file_list))
        exit(0)
    if (source.dir_in is None) or (not os.path.isdir(source.dir_in)):
        print("Input directory [%s] is not found." % (source.dir_in))
        exit(0)
    if (source.dir_out is None) or (not os.path.isdir(source.dir_out)):
        print("Output directory [%s] is not found." % (source.dir_out))
        exit(0)
    ##  preprocessing  ##
    Preprocessing(source, hypara)
    ##  exit  ##
    exit(1)


###  Network Setup  ###
model = Network(hypara).to(source.device)
# Network size
params = model.size()


###  Prediction Mode  ###
if source.mode == 'prediction':
    ##  check input
    if (source.pdb_in is None) or (not os.path.isfile(source.pdb_in)):
        print("Pdb file [%s] is not found." % (source.pdb_in))
        exit(0)
    if (source.param_in is None) or (not os.path.isfile(source.param_in)):
        print("Parameter file [%s] is not found." % (source.param_in))
        exit(0)
    ##  int code to amino-acid types
    i2aa = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
    ##  load pre-trained parameters
    model.load_state_dict(torch.load(source.param_in, map_location=torch.device(source.device)), strict=True)
    ##  input data setup
    dat1, dat2, dat3, label, mask, aa1 = pdb2input(source.pdb_in, hypara)
    dat1, dat2, dat3, label, mask = add_margin(dat1, dat2, dat3, label, mask, hypara.nneighbor)
    dat1 = torch.FloatTensor(dat1).squeeze().to(source.device)
    dat2 = torch.FloatTensor(dat2).squeeze().to(source.device)
    dat3 = torch.BoolTensor(dat3).squeeze().to(source.device)
    label = torch.LongTensor(label).squeeze().to(source.device)
    mask = torch.BoolTensor(mask).squeeze().to(source.device)
    ##  prediction
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



###  Training Mode  ###
if source.mode == 'training':
    ##  check input
    if (source.file_train is None) or (not os.path.isfile(source.file_train)):
        print("Training list [%s] is not found." % (source.file_train))
        exit(0)
    if (source.file_valid is None) or (not os.path.isfile(source.file_valid)):
        print("Validation list [%s] is not found." % (source.file_valid))
        exit(0)
    if (source.dir_in is None) or (not os.path.isdir(source.dir_in)):
        print("Directory [%s] is not found." % (source.dir_in))
        exit(0)
    if (source.onlypred is True) and (not os.path.isfile(source.param_in)):
        print("Parameter file [%s] is not found." % (source.param_in))
        exit(0)
        
    ##  weight initialization
    model.apply(weights_init)
    ##  For transfer learning
    if source.onlypred is True:
        model.load_state_dict(torch.load(source.param_in, map_location=torch.device(source.device)), strict=True)
        model.prediction.apply(weights_init)
    ##  dataloader setup
    train_dataset = BBGDataset(listfile=source.file_train, dir_in=source.dir_in, hypara=hypara)
    valid_dataset = BBGDataset(listfile=source.file_valid, dir_in=source.dir_in, hypara=hypara)
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True)
    ##  loss function
    criterion = nn.CrossEntropyLoss().to(source.device)
    ##  optimizer setup
    optimizer = RAdam(model.parameters(), lr=hypara.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hypara.nepoch-10, gamma=0.1)
    ##  training routine
    file = open(source.file_out, 'w')
    file.write("# Total Parameters : {:.2f}M\n".format(params/1000000))
    loss_min = float('inf')
    for iepoch in range(hypara.nepoch):
        loss_train, acc_train, loss_valid, acc_valid = float('inf'), 0, float('inf'), 0
        # training #
        loss_train, acc_train = train(model, criterion, source, train_loader, optimizer, hypara)
        # validation #
        loss_valid, acc_valid = valid(model, criterion, source, valid_loader)
        scheduler.step()
        file.write(' {epoch:3d}  LossTR: {loss_TR:.3f} AccTR: {acc_TR:.3f}  LossTS: {loss_TS:.3f} AccTS: {acc_TS:.3f}\n'
                   .format(epoch=iepoch+1, loss_TR=loss_train, acc_TR=acc_train, loss_TS=loss_valid, acc_TS=acc_valid))
        file.flush()
        # output params #
        if(loss_min > loss_valid):
            torch.save(model.state_dict(), source.param_out)
            loss_min = loss_valid


###  Test Mode  ###
if source.mode == 'test':
    ##  check input
    if (source.file_list is None) or (not os.path.isfile(source.file_list)):
        print("List file [%s] is not found." % (source.file_list))
        exit(0)
    if (source.dir_in is None) or (not os.path.isdir(source.dir_in)):
        print("Directory [%s] is not found." % (source.dir_in))
        exit(0)
    if (source.param_in is None) or (not os.path.isfile(source.param_in)):
        print("Parameter file [%s] is not found." % (source.param_in))
        exit(0)
    ##  load pre-trained parameters
    model.load_state_dict(torch.load(source.param_in, map_location=torch.device(source.device)), strict=True)
    ##  dataloader setup
    test_dataset = BBGDataset(listfile=source.file_list, dir_in=source.dir_in, hypara=hypara)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    ##  loss function
    criterion = nn.CrossEntropyLoss().to(source.device)
    ##  test
    loss_test, acc_test = float('inf'), 0
    loss_test, acc_test = test(model, criterion, source, test_loader)
    print("# Total: Loss: %5.3f  Acc: %6.2f %%" % (loss_test, acc_test))
