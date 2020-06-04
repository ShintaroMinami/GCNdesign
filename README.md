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
                    [--epochs [Int]] [--lr [Float]] [--in_pdb [File]]
                    [--in_list [File]] [--in_train_list [File]]
                    [--in_valid_list [File]] [--in_dir [Directory]]
                    [--out_dir [Directory]] [--out_resfile [File]]
                    [--in_params [File]] [--out_params [File]]
                    [--output [File]] [--device {cpu,cuda}] [--layer [Int]]
                    [--fragsize [Int]]

optional arguments:
  -h, --help            show this help message and exit
  --mode {prediction,training,test,preprocessing}
                        Running mode. (default:prediction)
  --epochs [Int]        Number of training epochs. (default:50)
  --lr [Float]          Learning rate. (default:0.002)
  --in_pdb [File]       PDB file input.
  --in_list [File]      List of data to be processed.
  --in_train_list [File]
                        List of training data.
  --in_valid_list [File]
                        List of validation data.
  --in_dir [Directory]  Directory in which data are stored.
  --out_dir [Directory]
                        Directory in which data processed will be stored.
  --out_resfile [File]  Resfile format file for RosettaDesign.
  --in_params [File]    Pre-trained parameter file.
                        (default:param_default.pkl)
  --out_params [File]   Trained parameter file. (default:"params_out.pkl")
  --output [File]       Output file. (default:"result.dat")
  --device {cpu,cuda}   Processing device.
  --layer [Int]         Number of GCN layers. (default:8)
  --fragsize [Int]      Fragment size of prediction module.(default:9)
```

## Author
* Shintaro Minami(https://github.com/ShintaroMinami)
* Nobuyasu Koga

## Citation
