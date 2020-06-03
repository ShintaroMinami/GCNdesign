# GCNdesign

A neural network model for prediction of amino-acid sequence from a protein backbone structure

## Requirement
* python3.x
* pytorch
* numpy
* pandas


## Usage
```
usage: GCNdesign.py [-h] [--mode {prediction,training,test,preprocessing}]
                    [--epochs EPOCHS] [--in_pdb IN_PDB] [--in_list IN_LIST]
                    [--in_train_list IN_TRAIN_LIST]
                    [--in_valid_list IN_VALID_LIST] [--in_dir IN_DIR]
                    [--out_dir OUT_DIR] [--out_resfile OUT_RESFILE]
                    [--in_params IN_PARAMS] [--out_params OUT_PARAMS]
                    [--output OUTPUT] [--device DEVICE] [--layer LAYER]

optional arguments:
  -h, --help            show this help message and exit
  --mode {prediction,training,test,preprocessing}
  --epochs EPOCHS
  --in_pdb IN_PDB
  --in_list IN_LIST
  --in_train_list IN_TRAIN_LIST
  --in_valid_list IN_VALID_LIST
  --in_dir IN_DIR
  --out_dir OUT_DIR
  --out_resfile OUT_RESFILE
  --in_params IN_PARAMS
  --out_params OUT_PARAMS
  --output OUTPUT
  --device DEVICE
  --layer LAYER
```

## Author
* Shintaro Minami(https://github.com/ShintaroMinami)
* Nobuyasu Koga
