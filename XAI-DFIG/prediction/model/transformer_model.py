#!/usr/bin/env python

# Import necessary libraries
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch import nn
from torch.nn import TransformerEncoderLayer
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from einops import rearrange
import numpy as np
import copy
from typing import Optional, Any, Union, Callable
import torch
import warnings
from torch import nn
from torch import Tensor
from torch.nn import functional as F
# ### Encoder layer


class MultiHeadAttention(nn.Module):
    """Implementation of basic MultiHeadAttention, see Attention Is All You Need paper
    (https://arxiv.org/abs/1706.03762) """

    # n_head: The number of attention heads.
    # dmodel: The dimensionality of the input and output.
    # d_k: The dimensionality of the key vectors.
    # d_v: The dimensionality of the value vectors.
    # d_hid: The dimensionality of the hidden layer in the feed-forward network (default is 2048).
    # dropout: The dropout probability for regularization (default is 0.1).
    # fixed: A flag indicating whether to use fixed random matrices for weight initialization (default is False).

    def __init__(self, n_head, dmodel, d_k, d_v, d_hid=2048, fixed=False):
        super().__init__()

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        dropout1 = 0.1

        self.n_head = n_head

        # Linear layers are used for projecting the input into query, key, and value spaces, respectively.
        # An instance of a linear transform is in form linear = nn.Linear(input_size, output_size).
        # linear transform takes an input tensor of shape (batch_size, input_size) and applies the weight and bias.
        # The resulting output tensor has a shape of (batch_size, output_size).
        #print('dmodel', dmodel)
        #print('n_head * d_k', n_head * d_k)

        self.w_qs = nn.Linear(dmodel, n_head * d_k)
        self.w_ks = nn.Linear(dmodel, n_head * d_k)
        self.w_vs = nn.Linear(dmodel, n_head * d_v)

        # If the fixed flag is True, these attributes hold pre-initialized fixed random matrices for weight initialization.
        self.w_qs_fixed = (torch.rand(dmodel, n_head * d_k) / np.sqrt(n_head * d_k)).to(device)
        self.w_ks_fixed = (torch.rand(dmodel, n_head * d_k) / np.sqrt(n_head * d_k)).to(device)
        self.w_vs_fixed = (torch.rand(dmodel, n_head * d_k) / np.sqrt(n_head * d_k)).to(device)


        # Initializes the weights of the linear layers using normal distribution initialization with specific standard deviation.
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (dmodel + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (dmodel + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (dmodel + d_v)))

        # Linear layers used in the feed-forward network component of the attention mechanism.
        # self.linear1 takes the input size of n_head * d_v and maps it to a hidden size of 1024.
        self.linear1 = nn.Linear(n_head * d_v, 1024)

        # self.linear2 takes the hidden size of 1024 and maps it back to the original input
        self.linear2 = nn.Linear(1024, dmodel)
        self.activation = F.relu

        # Dropout prevents overfitting in training by changing the elements of the input matrix zero with p probability.
        self.dropout = nn.Dropout(p=dropout1)

        # self.layer_norm is used to normalize the output of the attention mechanism
        self.layer_norm = nn.LayerNorm(dmodel)

        self.fixed = fixed

        # By setting self.batch_first = False, input tensors have a shape of (sequence_length, batch_size, feature_size).
        # when batch_first is True, the expected input shape becomes (batch_size, sequence_length, feature_size).
        self.batch_first = False

    # expects q, k, v  = b l d with b: batch, l: seq length and d: dimension of embedding - this is transposed to usual b d l convention
    # This is a result of projections being defined from the right (which require transposing when using self.w_qs = nn.Linear etc

    def forward(self, q, k, v, mask=None):
        residual = q

        # Using fixed projections
        if (self.fixed):

            # Multiplication of query and wq
            qs = torch.einsum('bij,jk->bik', q, self.w_qs_fixed)
            q = rearrange(qs, 'b l (head k) -> head b l k', head=self.n_head)

            # Multiplication of key and wk
            qk = torch.einsum('bij,jk->bik', k, self.w_ks_fixed)
            k = rearrange(qk, 'b l (head k) -> head b l k', head=self.n_head)

            # Multiplication of value and wv
            qv = torch.einsum('bij,jk->bik', v, self.w_vs_fixed)
            v = rearrange(qv, 'b l (head k) -> head b l k', head=self.n_head)

        # Using initialized weights
        else:
            #'It never goes here!', but it was in the initial implementation of transformers
            q = rearrange(self.w_qs(q), 'b l (head k) -> head b l k', head=self.n_head)
            k = rearrange(self.w_ks(k), 'b t (head k) -> head b t k', head=self.n_head)
            v = rearrange(self.w_vs(v), 'b t (head v) -> head b t v', head=self.n_head)

        # This line performs a dot product attention operation between the query tensor q and the key tensor k.
        # The resulting tensor is normalized by dividing it by the square root of the last dimension of the query tensor.

        attnraw = torch.einsum('hblk,hbtk->hblt', [q, k]) / np.sqrt(q.shape[-1])

        if mask is not None:
            # The mask[None] part adds a new dimension to the mask tensor to have the same shape as the attention tensor.
            # The operation fills the attention tensor elements with -inf at positions where the corresponding mask is True
            attnraw = attnraw.masked_fill(mask[None], -np.inf)

        # This line applies the softmax function along the fourth dimension of the attention tensor attn
        attn = torch.softmax(attnraw, dim=3)

        # Multiplication of attention and value
        attnallheads = torch.einsum('hblt,hbtv->hblv', [attn, v])
        output = rearrange(attnallheads, 'head b l v -> b l (head v)')

        # Applying two linear functions as the feed-forward neural network
        output = self.linear2(self.dropout(self.activation(self.linear1(output))))

        # Add with the output of the previous stage
        output = self.layer_norm(output + residual)
        return attnraw, output, attnallheads


class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # this attribute saves the value providedat object construction
        self.enable_nested_tensor = enable_nested_tensor
        # this attribute controls whether nested tensors are used
        self.use_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

        enc_layer = "encoder_layer"
        why_not_sparsity_fast_path = ''
        if not isinstance(encoder_layer, torch.nn.TransformerEncoderLayer):
            why_not_sparsity_fast_path = f"{enc_layer} was not TransformerEncoderLayer"
        elif encoder_layer.norm_first :
            why_not_sparsity_fast_path = f"{enc_layer}.norm_first was True"
        elif not encoder_layer.self_attn.batch_first:
            why_not_sparsity_fast_path = (f"{enc_layer}.self_attn.batch_first was not True" +
                                          "(use batch_first for better inference performance)")
        elif not encoder_layer.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn._qkv_same_embed_dim was not True"
        elif not encoder_layer.activation_relu_or_gelu:
            why_not_sparsity_fast_path = f"{enc_layer}.activation_relu_or_gelu was not True"
        elif not (encoder_layer.norm1.eps == encoder_layer.norm2.eps) :
            why_not_sparsity_fast_path = f"{enc_layer}.norm1.eps was not equal to {enc_layer}.norm2.eps"
        elif encoder_layer.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = f"{enc_layer}.self_attn.num_heads is odd"

        if enable_nested_tensor and why_not_sparsity_fast_path:
            warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
            self.use_nested_tensor = False

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)
        attnallheadslayers=[]
        #print('self.layers', len(self.layers))

        for mod in self.layers:
            outputlayer = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)
            #print('outputlayer',  len(outputlayer))
            attn_raw=outputlayer[0]
            output=outputlayer[1]
            attnallheads = outputlayer[2]

            #print('attnallheads', len(attnallheads))
            #print('output', len(output))

            attnallheadslayers.append(attnallheads)
        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())


        if self.norm is not None:
            output = self.norm(output)


        #print('attnallheads', len(attnallheadslayers))
        return [attn_raw, output, attnallheadslayers]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal
def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )

def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# In[36]:


# This class inherits from the TransformerEncoderLayer class, which is part of the PyTorch Transformer module.
class TransformerEncoderLayerCustom(TransformerEncoderLayer):

    def __init__(self, dmodel, nhead, d_hid=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False,
                 device=None, dtype=None):
        super().__init__(dmodel, nhead, d_hid, dropout,
                         activation, layer_norm_eps, batch_first, norm_first, device, dtype)
        d_k, d_v = dmodel, dmodel

        # Applying multi-head attention
        self.output =MultiHeadAttention(nhead, dmodel, d_k, d_v, d_hid=d_hid, fixed=False)
        self.activation = nn.ReLU()
        print("New MultiHeadAttention set with d_k=%s and d_v=%s" % (d_k, d_v))

    # expects src = b l d with b: batch, l: seq length and d: dimension of embedding - this is transposed to usual b d l convention
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        self.x = src
        self.x = self.output(self.x, self.x, self.x)
        return self.x


class TransformerModel(nn.Module):
    """Main TransformerModel used

    Attributes
    ----------
    dmodel : itensor
        input tensor
    seq_len : int
        length of sequence
    nhead : int
        number of heads in nn.MultiheadAttention
    d_hid : int
        dimension of the feedforward network model in nn.TransformerEncoder
    nlayers : int
        number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    """

    def __init__(self, dmodel, seq_len, nhead, d_hid, nlayers, num_classes, scale=10):
        super().__init__()
        self.dmodel = dmodel
        self.seq_len = seq_len
        self.scale = scale

        # Building transformer encoder layer
        encoder_layers = TransformerEncoderLayerCustom(dmodel, nhead, d_hid)
        print('encoderlayers',encoder_layers)

        self.wq= encoder_layers.output.w_qs.weight
        self.wk= encoder_layers.output.w_ks.weight
        self.wv= encoder_layers.output.w_vs.weight

        # Having several encoder layers
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # nn.linear applies a linear mapping to the input tensor by multiplying it with a weight matrix and adding bias
        # Input tensor has a flattened shape of (batch_size, seq_len * dmodel)
        # Output a tensor of shape (batch_size, num_classes)
        self.dropout = nn.Dropout(p=0.85)
        self.fc = nn.Linear(in_features=seq_len * dmodel, out_features=num_classes)

    def forward(self, x):
        # Need to transpose d and l to fit API of TransformerEncoderLayerCustom
        x = rearrange(x, 'n d l -> n l d')
        transformerencoder = self.transformer_encoder(x)
        attn_raw = transformerencoder[0]
        x0 = transformerencoder[1]
        attn_all_heads_layers= transformerencoder[2]
        x = rearrange(x0, 'n d l -> n (d l)')
        x = self.dropout(x)
        return self.fc(x), x0, attn_all_heads_layers,self.wq, self.wk, self.wv, attn_raw

class my_transformer_model(TransformerModel):
    def __init__(self, dmodel, seq_len, nhead, d_hid, nlayers, num_classes, scale=10):
        print("init called")
        super().__init__(dmodel, seq_len, nhead, d_hid, nlayers, num_classes, scale)
