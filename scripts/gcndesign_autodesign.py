#! /usr/bin/env python

from os import path
import sys
import argparse
import numpy as np

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('pdb', type=str, default=None, metavar='PDB File',
                    help='PDB file input.')
parser.add_argument('--nstruct', '-n', type=int, default=5, metavar='Int',
                    help='Number of structures output. (default:{})'.format(5))
parser.add_argument('--prefix', '-p', type=str, default='autodes', metavar='String',
                    help='Prefix for output PDB files. (default:{})'.format('autodes'))
parser.add_argument('--temperature', '-t', type=float, default=1.0, metavar='[Float]',
                    help='Temperature: probability P(AA) is proportional to exp(logit(AA)/T). (default:{})'.format(1.0))
parser.add_argument('--prob-cut', '-c', type=float, default=0.6, metavar='Float',
                    help='Probability cutoff. (default:{})'.format(0.6))
parser.add_argument('--scorefxn', '-s', type=str, default='ref2015', metavar='String',
                    help='Rosetta score function. (default:{})'.format('ref2015'))
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
parser.add_argument('--fastdesign-iterations', '-iter', type=int, default=2, metavar='Int',
                    help='Param "standard_repeats" for Rosetta FastDesign. (default:{})'.format(2))
parser.add_argument('--param-in', type=str, default=None, metavar='File',
                    help='NN parameter file. (default:{})'.format(None))
args = parser.parse_args()


# pyrosetta
try:
    import pyrosetta
except ModuleNotFoundError:
    print("PyRosetta is required for gcndesign_autodesign. [http://www.pyrosetta.org]")
    exit()
pyrosetta.init("-ignore_unrecognized_res 1 -ex1 -ex2aro")
scorefxn = pyrosetta.create_score_function(args.scorefxn)

# gcndesign predictor
dir_script = path.dirname(path.realpath(__file__))
sys.path.append(dir_script+'/../')
from gcndesign.predictor import Predictor
predictor = Predictor(param=args.param_in)

# pdb input
pose_in = pyrosetta.pose_from_pdb(args.pdb)
# max residue number
max_resnum = np.max([pose_in.pdb_info().number(i+1) for i in range(pose_in.size())])

## Setup TaskFactory
taskf = pyrosetta.rosetta.core.pack.task.TaskFactory()
taskf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
if args.include_init_restype:
    taskf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())

# resfile task-operation
from gcndesign.resfile import fix_native_resfile, expand_nums
resfile = predictor.make_resfile(pdb=args.pdb, prob_cut=args.prob_cut, unused=args.unused, temperature=args.temperature)
resfile = fix_native_resfile(resfile, resnums=expand_nums(args.keep, max_aa_num=max_resnum), keeptype=args.keep_type)
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
movemapf.all_jumps(setting=True)

## Check Setting
#display_pose = pyrosetta.rosetta.protocols.fold_from_loops.movers.DisplayPoseLabelsMover()
#display_pose.tasks(taskf)
#display_pose.movemap_factory(movemapf)
#display_pose.apply(pose)

## Mover Setting
fastdesign = pyrosetta.rosetta.protocols.denovo_design.movers.FastDesign(scorefxn_in=scorefxn, standard_repeats=args.fastdesign_iterations)
fastdesign.set_task_factory(taskf)
fastdesign.set_movemap_factory(movemapf)

## Apply
for i in range(args.nstruct):
    pose = pose_in.clone()
    fastdesign.apply(pose)
    file_out = '{:s}-{:03d}.pdb'.format(args.prefix, i+1)
    pose.dump_pdb(file_out)


