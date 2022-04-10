#! /usr/bin/env python

from os import path
import sys
import argparse
#import importlib

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('pdb', type=str, default=None, metavar='PDB File',
                    help='PDB file input.')
parser.add_argument('--nstruct', '-n', type=int, default=10, metavar='Int',
                    help='Number of structures output. (default:{})'.format(10))
parser.add_argument('--prefix', '-p', type=str, default='autodes', metavar='String',
                    help='Prefix for output PDB files. (default:{})'.format('autodes'))
parser.add_argument('--prob-cut', '-c', type=float, default=0.8, metavar='Float',
                    help='Probability cutoff. (default:{})'.format(0.8))
parser.add_argument('--scorefxn', '-s', type=str, default='ref2015', metavar='String',
                    help='Rosetta score function. (default:{})'.format('ref2015'))
parser.add_argument('--keep', '-k', type=str, default=[], metavar='Str', nargs='+',
                    help='Residue numbers for keeping the initial amino-acid type. (e.g. "-k 1 2 3 11-15 20-100 ...")')
parser.add_argument('--unused', '-u', type=str, default=None, metavar='Char', nargs='+',
                    help='Residue types not to be used in design sequences. (e.g. "-e C H W ...")')
parser.add_argument('--include-init-restype', default=False, action='store_true',
                    help='Include the initial residue type. (default:{})'.format(False))
parser.add_argument('--fastdesign-iterations', '-iter', type=int, default=1, metavar='Int',
                    help='Param "standard_repeats" for Rosetta FastDesign. (default:{})'.format(1))
parser.add_argument('--param-in', type=str, default=None, metavar='File',
                    help='NN parameter file. (default:{})'.format(None))
args = parser.parse_args()


# pyrosetta
try:
    import pyrosetta
except ModuleNotFoundError:
    print("PyRosetta is required for gcndesign_autodesign. [http://www.pyrosetta.org]")
    exit()
pyrosetta.init("-ignore_unrecognized_res 1 -ex1 -ex2aro -detect_disulf 0")
scorefxn = pyrosetta.create_score_function(args.scorefxn)

# gcndesign predictor
dir_script = path.dirname(path.realpath(__file__))
sys.path.append(dir_script+'/../')
from gcndesign.predictor import Predictor
predictor = Predictor(param=args.param_in)

# pdb input
pose_in = pyrosetta.pose_from_pdb(args.pdb)

## Setup TaskFactory
taskf = pyrosetta.rosetta.core.pack.task.TaskFactory()
taskf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
if args.include_init_restype:
    taskf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())

# resfile task-operation
from gcndesign.resfile import fix_native_resfile, expand_nums
resfile = predictor.make_resfile(pdb=args.pdb, prob_cut=args.prob_cut, unused=args.unused)
resfile = fix_native_resfile(resfile, resnums=expand_nums(args.keep))
readresfile = pyrosetta.rosetta.core.pack.task.operation.ReadResfile()
readresfile.set_cached_resfile(resfile)

# add readresfile to taskfactory
taskf.push_back(readresfile)

## Check TaskFactory Setting
packer_task = taskf.create_task_and_apply_taskoperations(pose_in)

## Setup MoveMapFactory
movemapf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
movemapf.all_bb(setting=True)
movemapf.all_chi(setting=True)
movemapf.all_jumps(setting=False)

## Check Setting
#display_pose = pyrosetta.rosetta.protocols.fold_from_loops.movers.DisplayPoseLabelsMover()
#display_pose.tasks(taskf)
#display_pose.movemap_factory(movemapf)
#display_pose.apply(pose)

## Mover Setting
fastdesign = pyrosetta.rosetta.protocols.denovo_design.movers.FastDesign(scorefxn_in=scorefxn, standard_repeats=1)
fastdesign.set_task_factory(taskf)
fastdesign.set_movemap_factory(movemapf)

## Apply
for i in range(args.nstruct):
    pose = pose_in.clone()
    fastdesign.apply(pose)
    file_out = '{:s}-{:03d}.pdb'.format(args.prefix, i+1)
    pose.dump_pdb(file_out)


