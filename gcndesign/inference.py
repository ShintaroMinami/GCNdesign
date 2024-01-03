import torch
from .models import GCNdesign
from .dataset import aa_types
from tqdm import tqdm
import numpy as np
from einops import repeat

# Gumbel Sampling
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)


class SequenceGenerator(GCNdesign):
    def __init__(self, checkpoint_file=None, device='cpu'):
        checkpoint = torch.load(checkpoint_file, map_location=device)
        model_hypara = checkpoint['model_hypara']
        super().__init__(**model_hypara)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(device)
        self.eval()
        self.device = device

    def resid_to_fasta(self, resid_batched):
        fasta = []
        for resid_list in resid_batched:
            fasta.append("".join([aa_types[i] for i in resid_list]))
        return fasta

    def autoregressive_generate(self, resid, node_feat, edge_feat, adjmat, temperature=0.1, num_replace=1):
        num_generate = (resid == self.mask_index).sum(-1)[0]
        for _ in tqdm(range(num_generate//num_replace + 1)):
            generated_now = resid[:,:] != self.mask_index
            num_replace_now = min(num_generate - generated_now.sum(-1)[0].item(), num_replace)
            logits, _ = self.decode(node_feat, edge_feat, adjmat, masked_resid=resid)
            pred_idx = gumbel_sample(logits, temperature=temperature, dim=-1)
            occupied_now = (resid != self.mask_index).to(bool)
            mutate_place = torch.zeros_like(occupied_now).to(bool)
            for batch_idx, occ in enumerate(occupied_now.cpu()):
                candidates = np.where(occ == False)[0]
                res_idx = candidates if len(candidates) == num_replace_now else np.random.choice(candidates, num_replace_now, replace=False)
                mutate_place[batch_idx, res_idx] = True
            # update residues
            resid = resid * ~mutate_place + pred_idx * mutate_place
        # return
        return resid

    @torch.no_grad()
    def design_sequences(self, input_pdb, num_sequences=10, temperature=0.1, num_replace=1):
        # Encode
        node, edge, adjmat = self.process_pdbfile(input_pdb)
        node, edge, adjmat = node, edge, adjmat.squeeze()
        node = repeat(node, 'l c -> b l c', b=num_sequences)
        edge = repeat(edge, 'l1 l2 c -> b l1 l2 c', b=num_sequences)
        adjmat = repeat(adjmat, 'l1 l2 -> b l1 l2', b=num_sequences)
        node, edge = self.encode(node, edge, adjmat)
        B, L, _ = node.shape
        # Initial Sequence
        resid_init = torch.ones([B, L], dtype=int, device=self.device) * self.mask_index
        # Autoregressive Generation
        resid = self.autoregressive_generate(
            resid_init, node, edge, adjmat,
            temperature=temperature, num_replace=num_replace)
        # to Fasta
        fasta = self.resid_to_fasta(resid)
        return fasta
        
        

