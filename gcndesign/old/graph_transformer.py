import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch import nn, einsum
from rotary_embedding_torch import RotaryEmbedding

from einops.layers.torch import Rearrange
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

@contextmanager
def disable_tf32():
    orig_value = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = orig_value


# classes

def FeedForward(dim, mult=1., num_layers=2, act=nn.ReLU):
    layers = []
    dim_hidden = dim * mult
    for ind in range(num_layers):
        is_first = ind == 0
        is_last  = ind == (num_layers - 1)
        dim_in   = dim if is_first else dim_hidden
        dim_out  = dim if is_last else dim_hidden
        layers.append(nn.Linear(dim_in, dim_out))
        if is_last:
            continue
        layers.append(act())
    return nn.Sequential(*layers)


class GraphAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        key_dim = 16,
        value_dim = 16,
        dim_edge = None,
        use_rotary_embedding = True,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.heads = heads
        # num attention contributions
        num_attn_logits = 2
        # qkv projection for attention (normal)
        self.attn_logits_scale = (num_attn_logits * key_dim) ** -0.5
        self.to_q = nn.Linear(dim, key_dim * heads, bias = False)
        self.to_k = nn.Linear(dim, key_dim * heads, bias = False)
        self.to_v = nn.Linear(dim, value_dim * heads, bias = False)
        # rotary embedding
        self.rotary_emb = RotaryEmbedding(dim=key_dim)
        # edge representation projection to attention bias
        dim_edge = default(dim_edge, dim)
        self.edge_attn_logits_scale = num_attn_logits ** -0.5
        self.to_edge_attn_bias = nn.Sequential(
            nn.Linear(dim_edge, heads),
            Rearrange('b ... h -> (b h) ...')
        )
        # combine out - node + edge dim
        self.to_out = nn.Linear(heads * (value_dim + dim_edge), dim)

    def forward(
        self,
        node_in,
        edge_in,
        mask = None
    ):
        x, e, h = node_in, edge_in, self.heads
        # get queries, keys, values for attention
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        # split out heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        # rotary embedding
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        # derive attn logits for scalar and pairwise
        attn_logits_node = einsum('b i d, b j d -> b i j', q, k) * self.attn_logits_scale
        attn_logits_edge = self.to_edge_attn_bias(e) * self.edge_attn_logits_scale
        # combine attn logits
        attn_logits = attn_logits_node + attn_logits_edge
        # mask
        if exists(mask):
            mask = rearrange(mask, 'b i -> b i ()') * rearrange(mask, 'b j -> b () j')
            mask = repeat(mask, 'b i j -> (b h) i j', h = h)
            mask_value = max_neg_value(attn_logits)
            attn_logits = attn_logits.masked_fill(~mask, mask_value)
        # attention
        attn = attn_logits.softmax(dim = - 1)
        with disable_tf32(), autocast(enabled = False):
            # disable TF32 for precision
            results_node = einsum('b i j, b j d -> b i d', attn, v)
            attn_with_heads = rearrange(attn, '(b h) i j -> b h i j', h = h)
            results_edge = einsum('b h i j, b i j d -> b h i d', attn_with_heads, e)
        # merge back heads
        results_node = rearrange(results_node, '(b h) n d -> b n (h d)', h = h)
        results_edge = rearrange(results_edge, 'b h n d -> b n (h d)', h = h)
        # concat results and project out
        results = torch.cat([results_node, results_edge], dim = -1)
        return self.to_out(results)


# one transformer block
class GTBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_edge = None,
        ff_mult = 1,
        ff_num_layers = 3,
        post_norm = True,
        post_attn_dropout = 0.,
        post_ff_dropout = 0.,
        update_edge = True,
        **kwargs
    ):
        super().__init__()
        self.update_edge = update_edge
        self.post_norm = post_norm
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = GraphAttention(dim=dim, **kwargs)
        self.post_attn_dropout = nn.Dropout(post_attn_dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mult=ff_mult, num_layers=ff_num_layers)
        self.post_ff_dropout = nn.Dropout(post_ff_dropout)
        if update_edge:
            dim_edge = default(dim_edge, dim)
            self.edge_reshape = nn.Linear(dim_edge+dim*2, dim_edge)
            self.edge_norm = nn.LayerNorm(dim_edge)
            self.edge_ff = FeedForward(dim_edge, mult=ff_mult, num_layers=ff_num_layers)
            self.edge_dropout = nn.Dropout(post_ff_dropout)

    def forward(self, x, e, **kwargs):
        post_norm = self.post_norm
        attn_input = x if post_norm else self.attn_norm(x)
        x = self.attn(attn_input, e, **kwargs) + x
        x = self.post_attn_dropout(x)
        x = self.attn_norm(x) if post_norm else x
        ff_input = x if post_norm else self.ff_norm(x)
        x = self.ff(ff_input) + x
        x = self.post_ff_dropout(x)
        x = self.ff_norm(x) if post_norm else x
        if self.update_edge:
            e = e if post_norm else self.edge_norm(e)
            x_expand = repeat(x, 'b n d -> b n m d', m=x.shape[1])
            e_r = self.edge_reshape(torch.cat([e, x_expand, x_expand.transpose(1,2)], dim=-1))
            e = self.edge_ff(e_r) + e
            e = self.edge_norm(e) if post_norm else e
        return x, e


class GraphTransformer(nn.Module):
    def __init__(
        self,
        dim_node: int=128,
        dim_edge: int=64,
        depth: int=6,
        nneighbor: int=None,
        r_drop: float=0.2,
        **kwargs
    ):
        super().__init__()
        # layers
        self.dim_node = dim_node
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                GTBlock(dim=dim_node, dim_edge=dim_edge, post_attn_dropout=r_drop, post_ff_dropout=r_drop, **kwargs),
                None
            ]))

    def forward(
        self,
        node_in,
        edge_in,
        adjmat_in,
    ):
        x, e = node_in, edge_in
        b, n, *_ = node_in.shape
        # go through the layers and apply invariant point attention and feedforward
        for update_node, _ in self.layers:
            x, e = update_node(x, e)
        return x, e