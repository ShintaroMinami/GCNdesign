# GCNdesign

A neural network model for prediction of amino-acid probability from a protein backbone structure.

### Built with
- pytorch
- numpy
- pandas
- tqdm

## Getting Started

### Install
To install gcndesgn through [pip](https://pypi.org/project/gcndesign)
```bash
pip install gcndesign
```

## Usage

### Quick usage as a python module
```python
from gcndesign.prediction import Predictor

gcndes = Predictor(device='cpu') # 'cuda' can also be applied
gcndes.pred(pdb='pdb-file-path') # returns list of amino-acid probabilities
```

### Usage of scripts

```gcndesign_predict.py```

To predict amino-acid probabilities for each residue-site
```bash
gcndesign_predict  YOUR_BACKBONE_STR.pdb
```

```gcndesign_autodesign.py```

To design 20 sequences in a completely automatic fashion

```bash
gcndesign_autodesign  YOUR_BACKBONE_STR.pdb  -n 20
```

For more detailed usage, please run the following command
```bash
gcndesign_autodesign -h
```

> note
>
> The gcndesign_autodesign script requires **pyrosetta** software.
> Installation & use of **pyrosetta** must be in accordance with their license.

## External Packages
- gcndesign_autodesign.py: [**PyRosetta**](https://www.pyrosetta.org/)

## Issues
This code is not completely compatible with an input of a protein complex structure.

## Lisence
Distributed under [MIT](https://choosealicense.com/licenses/mit/) license.

## Acknowledgments
The author was supported by Grant-in-Aid for JSPS Research Fellows (PD, 17J02339).
Koga Laboratory of Institute for Molecular Science has provided a part of the computational resources.
Koya Sakuma ([yakomaxa](https://github.com/yakomaxa)) gave a critical idea for neuralnet architecture design in a lot of deep discussions.

