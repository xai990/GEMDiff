
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from .nn import timestep_embedding
from . import logger
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

"""
2024/03/01
log:

* add the position embedding 

2024/03/05
log:
* set `is_causal=False `

log:
* add assert function to make sure the gene feature can be divided by patch size

2024/03/08
log:
* change layernorm elementwise_affine True -> False 
* MLP GELU approximate none -> tanh
* unpatch sequence rearange 
* double layernorm -> single layernorm 
still has a double layernorm 
"""



"""
2024/03/12

log:
* change back to original code 
* 1st add LN in SPT ( LN does not work)
* 2nd MLP GELU approximate none -> tanh
* 3rd pos embd 
2 + 3 works 


2024/03/22
log:
* MLP GELU approximate none -> tanh does not perform well 
* change it back to none 

2024/04/02
log:
* learn sigma 

"""


class CausalSelfAttention(nn.Module):

    def __init__(self,
        n_embd,
        n_head,
        dropout,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # logger.debug(f"The size of q is: {q} -- mlp")
        # logger.debug(f"The size of k is: {k} -- mlp")
        # logger.debug(f"The size of v is: {v} -- mlp")
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class MLP(nn.Module):

    def __init__(self,
        n_embd,
        dropout,
    ):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd)
        # self.gelu    = nn.GELU(approximate="tanh") # change 2nd
        self.gelu    = nn.GELU() 
        self.c_proj  = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    
class Block(nn.Module):

    def __init__(self,
        n_embd,
        n_head,
        dropout,
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.attn = CausalSelfAttention(n_embd,n_head,dropout)
        self.ln_2 = nn.LayerNorm(n_embd, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    
class SPT(nn.Module):
    def __init__(self, *, dim=768,patch_size):
        super().__init__()
        patch_dim = patch_size
        self.to_patch_tokens = nn.Sequential(
            Rearrange('b (l p) -> b l p', p = patch_size),
            # nn.LayerNorm(patch_dim), # 3.12 change 1 
            nn.Linear(patch_dim, dim),
            
        )
    
    def forward(self,x):
        return self.to_patch_tokens(x)

    
class RSPT(nn.Module):
    def __init__(self, *, n_embd=768,patch_dim,):
        super().__init__() 
        self.back_patch_tokens = nn.Sequential(
            nn.GELU(), 
            nn.Linear(n_embd, patch_dim),
            Rearrange('b l p -> b (p l)', p = patch_dim),
        )
    
    def forward(self,x):
        return self.back_patch_tokens(x)    

        
class GPT(nn.Module):

    def __init__(self,
        gene_feature,
        patch_size,
        n_head,
        dropout,
        n_embd=768,
        n_layer=4,
        num_classes = None,
        learn_sigma = True,
    ):
        super().__init__()
        # assert config.vocab_size is not None
        # assert config.block_size is not None
        # self.config = config
        self.t_embd = n_embd
        self.learn_sigma = learn_sigma 
        self.gene_feature = gene_feature
        self.time_embed = nn.Sequential(
            nn.Linear(n_embd, self.t_embd*4),
            nn.SiLU(),
            nn.Linear(self.t_embd*4, self.t_embd),
        )
        self.num_classes = num_classes
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, self.t_embd)
        # self.gte = nn.Linear(1000,n_embd)
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Linear(gene_feature, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(n_embd,n_head,dropout) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd),
        ))
        self.patch_size = patch_size
        num_patches = (gene_feature // patch_size)
        self.num_patches = num_patches 
        self.patch_dim_out = patch_size * 2 if learn_sigma else patch_size 
        self.to_patch_embedding = SPT(dim = n_embd, patch_size = patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, n_embd)) # change 3rd pos 
        self.back_patch_embedding = RSPT(n_embd = n_embd, patch_dim = self.patch_dim_out)
        # self.transformer = nn.ModuleDict(dict(
        #     wte = nn.Embedding(config.vocab_size, config.n_embd),
        #     wpe = nn.Embedding(config.block_size, config.n_embd),
        #     drop = nn.Dropout(config.dropout),
        #     h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        #     ln_f = nn.LayerNorm(config.n_embd),
        # ))
        
        # self.lm_head = nn.Linear(n_embd, gene_feature, bias=False)
        self.lm_head = nn.Identity()
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.gte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, timesteps, y=None):
        # device = idx.device
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        emb = self.time_embed(timestep_embedding(timesteps, self.t_embd))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)
        
        # b, *features = x.size()
        # logger.debug(f"The size of x is:{x.size()} --mlp")
        assert (x.shape[-1] % self.patch_size == 0), " gene feature must be an integer multiple of patch size"
        x = self.to_patch_embedding(x)
        _,n,_ = x.shape
        # logger.debug(f"The size of number patch is: {self.num_patches} --mlp")
        # logger.debug(f"The number of patch is is:{self.num_patches} -- mlp")
        # logger.debug(f"The embed shape is:{emb.size()} -- mlp")
        
        x += self.pos_embedding # change 3rd pos 
        # x = self.dropout(x)
        # h = x.view(b, self.gene_block, -1)
        
        # assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        # logger.debug(f"The h size is:{h.size()} -- mlp")
        # logger.debug(f"The gene block size is:{self.gene_block} -- unet")
        # logger.debug(f"The gene feature is:{self.gene_feature} -- unet")
        # logger.debug(f"The t embedding is:{self.t_embd} -- unet")
        # forward the GPT model itself
        # tok_emb = self.transformer.wte(h)
        # tok_emb = self.gte(h) # token embeddings of shape (b, t, n_embd)
        # pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        # x = self.transformer.drop(tok_emb + pos_emb)
        # add the time 
    
        x = self.transformer.drop(x + emb.unsqueeze(1).expand(-1, self.num_patches, -1))
        # logger.debug(f"The x shape is:{x.size()} -- mlp")
        for block in self.transformer.h:
            # logger.debug(f"The block is: {block} -- unet")
            x = block(x)
        x = self.transformer.ln_f(x)
        # logger.debug(f"The size of x is: {x.size()} -- unet")
        # if targets is not None:
        #     # if we are given some desired targets also calculate the loss
        #     logits = self.lm_head(x)
        #     # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        # else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
        x = self.back_patch_embedding(x)
        out = self.lm_head(x) # note: using list [-1] to preserve the time dim
            # loss = None
        # logger.debug(f"The size of logits:{logits.size()} -- unet")
        return out