# GCNdesign

A neural network model for prediction of amino-acid probability from a protein backbone structure.

## Installation
Use the package manager [pip](https://pypi.org/project/gcndesign) to install gcndesign.
```bash
pip install gcndesign
```

## Quick usage as a python module
```python
from gcndesign.prediction import Predictor

gcndes = Predictor(device='cpu') # 'cuda' can also be applied
gcndes.pred(pdb='pdb-file-path') # returns list of amino-acid probabilities
```

## Usage of scripts

### gcndesign_predict
To predict amino-acid probabilities for each residue-site
```bash
gcndesign_predict  YOUR_BACKBONE_STR.pdb
```

### gcndesign_autodesign
To design 20 sequences in a completely automatic fashion

```bash
gcndesign_autodesign  YOUR_BACKBONE_STR.pdb  -n 20
```

For more detailed usage, please run the following command
```bash
gcndesign_autodesign -h
```

* The gcndesign_autodesign script requires **pyrosetta** software.
* Installation & use of **pyrosetta** must be in accordance with their license.



## Author
* Shintaro Minami (https://github.com/ShintaroMinami)

## Lisence
[MIT](https://choosealicense.com/licenses/mit/)