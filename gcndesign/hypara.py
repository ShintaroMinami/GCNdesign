import dataclasses
from os import path
dir_params = path.dirname(path.realpath(__file__))

##  Hyper Parameters  ##
@dataclasses.dataclass
class HyperParam:
    nepoch:             int = 90
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
    niter_embed_rgc:    int =   5
    k_node_rgc:         int =  20
    k_edge_rgc:         int =  20
    d_embed_h_node:     int = 256
    d_embed_h_edge:     int = 256
    nlayer_embed_node:  int =   2
    nlayer_embed_edge:  int =   2
    # for prediction layer #
    fragment_size:      int =   9
    d_pred_h1:          int = 128
    d_pred_h2:          int =  64
    nlayer_pred:        int =   4


##  Input  ##
@dataclasses.dataclass
class InputSource:
    pdb_in:      str = None
    file_list:   str = None
    file_train:  str = None
    file_valid:  str = None
    file_out:    str = 'result.dat'
    dir_out:     str = None
    dir_in:      str = None
    param_prefix:str = 'params_out'
    param_in:    str = path.join(dir_params, 'params/param_default.pkl')
    onlypred:   bool = False
    resfile_out: str = None
    prob_cut:  float = 0.80
    device:      str = 'cpu'
