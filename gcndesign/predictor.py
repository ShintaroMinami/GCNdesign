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
aa2i = {aa:i for i, aa in enumerate(i2aa)}

def eliminate_restype(prob, unused):
    mask = np.ones_like(prob, dtype=bool)
    for aa in unused:
        mask[:, aa2i[aa]] = False
    # mask non-used residue types
    prob = prob * mask
    # normalize
    prob = prob / np.repeat(prob.sum(axis=-1)[:,None], prob.shape[-1], axis=-1)
    return prob

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
        assert path.isfile(self.param), "Parameter file {:s} was not found.".format(self.param)
        self.model = torch.load(self.param, map_location=torch.device(self.device))
        return

    def _pred_base(self, pdb: str):
        # input data setup
        dat1, dat2, dat3, label, mask, aa1 = pdb2input(pdb, self.hypara)
        dat1, dat2, dat3, label, mask = add_margin(dat1, dat2, dat3, label, mask, self.hypara.nneighbor)
        dat1 = dat1.squeeze().to(self.device)
        dat2 = dat2.squeeze().to(self.device)
        dat3 = dat3.squeeze().to(self.device)
        label = label.squeeze().to(self.device)
        mask = mask.squeeze().to(self.device)
        # prediction
        self.model.eval()
        outputs = self.model(dat1, dat2, dat3)[1:-1]
        # return
        return outputs, aa1

    def predict_logit_tensor(self, pdb: str, as_dict=False):
        # check pdb file
        assert path.isfile(pdb), "PDB file {:s} was not found.".format(pdb)
        # pred
        logit, _ = self._pred_base(pdb)
        logit = logit.detach().cpu().numpy()
        # return summary
        return [dict(zip(i2aa, l)) for l in logit] if as_dict else logit

    def predict(self, pdb: str, temperature: float=1.0):
        # check pdb file
        assert path.isfile(pdb), "PDB file {:s} was not found.".format(pdb)
        # original resnum
        pbb = ProteinBackbone(file=pdb)
        id2org = [(int(v[1:]), v[0]) for v in pbb.iaa2org]
        # pred
        logit, aa1 = self._pred_base(pdb)
        # convert to probabiality
        prob = torch.softmax(logit/temperature, dim=1).detach().cpu().numpy()
        # return summary
        pdict = [dict(zip(i2aa, p)) for p in prob]
        return [(p, {'resnum':v[0],'chain':v[1],'original':a}) for p,v,a in zip(pdict, id2org, aa1)]

    def make_resfile(self, pdb: str, temperature: float=1.0, prob_cut: float=0.8, unused=None):
        # check pdb file
        assert path.isfile(pdb), "PDB file {:s} was not found.".format(pdb)
        # restypes not to be used
        unused = [] if unused==None else unused
        # original resnum
        pbb = ProteinBackbone(file=pdb)
        id2org = [(int(v[1:]), v[0]) for v in pbb.iaa2org]
        # pred
        logit, aa1 = self._pred_base(pdb)
        # convert to probabiality
        prob = torch.softmax(logit/temperature, dim=1).detach().cpu().numpy()
        # eliminate non-used restypes
        prob = eliminate_restype(prob, unused)
        # resfile
        prob = [(p, *v) for p,v in zip(prob, id2org)]
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
