Using Triton-based layers
=========================

Triton_ is a language and compiler for parallel programming, currently applicable to CUDA-enabled GPUs.
It is compatible with PyTorch CUDA Tensors, and can be interfaced directly with pure python code.


PyTorch provides many primitives capable of tranforming tensors, which correspond to operators in each of the supported backends.
There are limits to how many of them can be supported at any point in time, short of supporting a JIT toolchain,
so some operations typical of the Transformer family are supported in PyTorch as a sequence of base operators.

Triton makes it possible to consolidate some of them into ad-hoc fused operators, which are compiled just-in-time.
xFormers proposes a couple of optimized layers, and a goal is to increase their number  and qualityover time.


Fused softmax layer
-------------------

This is a drop-in replacement to `torch.nn.softmax`_, the only limitation being that the softmax operation is limited to the last dimension.
Log-softmax is also available. The actual Triton kernel is very similar to `this tutorial<https://triton-lang.org/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py>`
The causal option improves the performance if the attention is causal to begin with, but in that case the blocksparse attention (simply using a lower triangular layout)
would be even better in terms of speed and memory use.


.. code-block:: python

    from xformers.triton import softmax, log_softmax
    y = softmax(x)   # Torch AMP, autograd aware


These curves can be replicated locally by running

.. code-block:: bash

    python3 xformers/benchmarks/benchmark_triton_softmax.py


The following curves have been collected on a V100, Triton 1.1 and PyTorch 1.9. At very large sizes and with this hardware Triton can currently suffer from register spilling,
using Ampere GPUs moves this issue further down the line.


.. figure:: ../../plots/Softmax_Bandwidth_FW_fp16.png

.. figure:: ../../plots/Softmax_Bandwidth_FW_BW_fp16.png

.. figure:: ../../plots/Softmax_Bandwidth_FW_fp32.png

.. figure:: ../../plots/Softmax_Bandwidth_FW_BW_fp16.png


Fused linear layer
-------------------
This is a drop-in replacement to two PyTorch operands: a `torch.nn.Linear`, with or without bias, and an activation, like `torch.nn.ReLU`. It is Torch AMP and autograd aware, and can be used very simply:

.. code-block:: python

    from xformers.triton import FusedLinearLayer
    my_linear_layer = FusedLinearLayer(in_features, out_features, bias=True/False, activation="squared_relu")
    ...
    y = my_linear_layer(x)

It is possible to skip either the bias or the activation (just use `None` in that case). As of September 2021, this layer is **faster than PyTorch for inference, non-sigmoid activations and fp16**.
In all other usecases, you will be better served using PyTorch. This is a work in progress and this could change over time.

The following is an example of the measured performance on a nVidia V100.

+-----------------------------------------+----------------------+--------------------+--------------------+----------------------+--------------------+
| torch.float16                           | Unit: TFlops         |                    |                    |                      |                    |
+=========================================+======================+====================+====================+======================+====================+
|                                         | B=8, M=256, K=512    | B=8, M=512, K=1024 | B=4, M=1024, K=1024| B=2, M=2048, K=2048  | B=2, M=4096, K=4096|
+-----------------------------------------+----------------------+--------------------+--------------------+----------------------+--------------------+
| pytorch - squared_relu -  bias - fw     |     6.3              |    12.4            |     12.3           |      17.1            |     19.0           |
+-----------------------------------------+----------------------+--------------------+--------------------+----------------------+--------------------+
| triton  - squared_relu -  bias - fw     |     13.8             |    18.9            |     18.9           |      21.9            |     21.7           |
+-----------------------------------------+----------------------+--------------------+--------------------+----------------------+--------------------+


.. _Triton: https://triton-lang.org/
.. _`torch.nn.softmax`: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html


Fused layer norm
----------------
This is a mostly drop-in replacement to PyTorch LayerNorm, although it will only process the last dimension.

.. code-block:: python

    from xformers.triton import FusedLayerNorm
    layernorm = FusedLayerNorm(X.shape[-1], eps).to("cuda")
    ...
    y = layernorm(X_)


These curves can be replicated locally by running

.. code-block:: bash

    python3 xformers/benchmarks/benchmark_triton_layernorm.py


The following curves have been collected on a V100, Triton 1.1 and PyTorch 1.9. At very large sizes and with this hardware Triton can currently suffer from register spilling,
using Ampere GPUs moves this issue further down the line.

.. figure:: ../../plots/LayerNorm_FW_torch.float16.png

.. figure:: ../../plots/LayerNorm_FW+BW_torch.float16.png

.. figure:: ../../plots/LayerNorm_FW_torch.float32.png

.. figure:: ../../plots/LayerNorm_FW+BW_torch.float32.png


Blocksparse
------------
This uses the blocksparse computations provided by Triton_ out of the box. In xFormers, you can use the BlocksSparseAttention,
which will limit the computations and the memory to the tiles specified in the `layout`:

.. code-block:: python

    import torch
    from xformers.components.attention import BlockSparseAttention
    BLOCK_SIZE = 32     # for instance
    SEQ = 1024          # ..
    layout = torch.tril(torch.ones(SEQ//BLOCK_SIZE, SEQ//BLOCK_SIZE)).cuda()  # Only compute the lower triangular
    lower_tril_attention = BlockSparseAttention(layout=layout, block_size=BLOCK_SIZE, dropout=0.1)
    causal_mask = torch.tril(torch.ones(SEQ, SEQ)).bool().cuda()
    ...
    att = lower_tril_attention(k, q, v, att_mask=causal_mask)


One way to understand the layout field is to think about the attention matrix as being tiled, the size of the tiles being specified by `block_size`.
The layout field then is a positive marker of the tiles which should be computed (ie: a layout being equal to torch.ones() means that everything is computed).
