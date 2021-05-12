from os import path
import numpy as np
import torch
from .hypara import HyperParam, InputSource
from .models import GCNdesign
from .dataset import pdb2input, add_margin
from .pdbutil import ProteinBackbone

# int code to amino-acid types
i2aa = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')

# for default paramfile
source = InputSource()

class Predictor():
    def __init__(self, device: str=None, param: str=None, hypara=None):
        # device
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hypara = hypara if hypara else HyperParam()
        self.param = param if param else InputSource().param_in
        self.device = device
        # model setup
        self.model = GCNdesign(self.hypara).to(self.device)
        # load pre-trained parameters
        assert path.isfile(self.param), "Parameter file {:s} was not found.".format(self.param)
        self.model.load_state_dict(torch.load(self.param, map_location=torch.device(self.device)), strict=True)

    def _pred_base(self, pdb: str):
        # input data setup
        dat1, dat2, dat3, label, mask, aa1 = pdb2input(pdb, self.hypara)
        dat1, dat2, dat3, label, mask = add_margin(dat1, dat2, dat3, label, mask, self.hypara.nneighbor)
        dat1 = torch.FloatTensor(dat1).squeeze().to(self.device)
        dat2 = torch.FloatTensor(dat2).squeeze().to(self.device)
        dat3 = torch.BoolTensor(dat3).squeeze().to(self.device)
        label = torch.LongTensor(label).squeeze().to(self.device)
        mask = torch.BoolTensor(mask).squeeze().to(self.device)
        # prediction
        self.model.eval()
        outputs = self.model(dat1, dat2, dat3)
        prob = torch.softmax(outputs, dim=1).detach().numpy()[1:-1]
        # return
        return prob, aa1

    def predict(self, pdb: str):
        # check pdb file
        assert path.isfile(pdb), "PDB file {:s} was not found.".format(pdb)
        # original resnum
        pbb = ProteinBackbone(file=pdb)
        id2org = [(int(v[1:]), v[0]) for v in pbb.iaa2org]
        # pred
        prob, aa1 = self._pred_base(pdb)
        # return summary
        pdict = [dict(zip(i2aa, p)) for p in prob]
        return [(p, {'resnum':v[0],'chain':v[1],'original':a}) for p,v,a in zip(pdict, id2org, aa1)]

    def make_resfile(self, pdb: str, prob_cut: float=0.8):
        # check pdb file
        assert path.isfile(pdb), "PDB file {:s} was not found.".format(pdb)
        # original resnum
        pbb = ProteinBackbone(file=pdb)
        id2org = [(int(v[1:]), v[0]) for v in pbb.iaa2org]
        # pred
        prob, aa1 = self._pred_base(pdb)
        prob = [(p, *v) for p,v in zip(prob, id2org)]
        # resfile
        line_resfile = 'start\n'
        for id,(p,i,a) in enumerate(prob):
            line_resfile += ' {:4d} {:s} PIKAA  '.format(i, a)
            pikaa = ''
            psum = 0
            sorted_args = np.argsort(-p)
            for i in range(20):
                iarg = sorted_args[i]
                pikaa = pikaa + i2aa[iarg]
                psum += p[iarg]
                if i > 0 and psum > prob_cut: break
            line_resfile += '{:20s} # {:s}\n'.format(pikaa, aa1[id])
        # return
        return line_resfile