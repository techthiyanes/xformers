See sparsity in action
######################

There is nothing specific to be done, as long as you are using the normal `scaled_dot_product` attention !
As soon as it is called with a sparse enough mask (`density < 30%`), then the computations will be sparse,
thanks to a variant of the Sputnik_ or GE-SpMM_ kernels.

.. code-block:: python

    import torch
    from xformers.components.attention import ScaledDotProduct

    attention = ScaledDotProduct().cuda()

    # FW a random bunch of data
    inputs = torch.rand((16, 1024, 1024), device=torch.device("cuda"))

    # Not a very sparse mask to begin with
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mask = (torch.rand((1024, 1024)) < 0.9).cuda()
    att = attention(q=inputs, k=inputs, v=inputs, mask=mask)

    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
    print(f"Dense - Peak memory use: {max_memory}MB")

    # Now use a very sparse mask and observe that memory use changes
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mask = (torch.rand((1024, 1024)) < 0.1).cuda()
    att = attention(q=inputs, k=inputs, v=inputs, mask=mask)

    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
    print(f"Sparse - Peak memory use: {max_memory}MB")


You should see something along the lines of:

.. code-block:: bash

    Dense - Peak memory use: 501MB
    Sparse - Peak memory use: 321MB


And voilà ! Note that we did not change anything in the model,
the input mask was just sparse enough to trigger the specific CUDA kernels.

.. _Sputnik: https://github.com/google-research/sputnik
.. _GE-SpMM: https://github.com/hgyhungry/ge-spmm

Replace all attentions from an existing ViT model with a sparse equivalent ?
############################################################################

Let's say you're used to working with a given Transformer based model, and want to experiment with one of the attention mechanisms supported by xFormers.

The following example shows how to do that in a particular example (reusing a reference ViT from pytorch-image-models_), but some aspects will translate just as well
when considering other model sources. In any case, please check the notebooks in the repository_ for a more exhaustive take.


.. code-block:: python

    import timm
    from timm.models.vision_transformer import VisionTransformer
    from xformers.components.attention import ScaledDotProduct
    from xformers.components.attention.helpers import TimmAttentionWrapper
    img_size = 224
    patch_size = 16

    # Get a reference ViT model
    model = VisionTransformer(img_size=img_size, patch_size=patch_size,
                                embed_dim=96, depth=8, num_heads=8, mlp_ratio=3.,
                                qkv_bias=False, norm_layer=nn.LayerNorm).cuda()


    # Define the mask that we want to use
    # We suppose in this snipper that you have a precise mask in mind already
    # but several helpers and examples are proposed in  `xformers.components.attention.attention_patterns`
    my_fancy_mask : torch.Tensor  # This would be for you to define

    # Define a recursive monkey patching function
    def replace_attn_with_xformers_one(module, att_mask):
        module_output = module
        if isinstance(module, timm.models.vision_transformer.Attention):
            qkv = module.qkv
            dim = qkv.weight.shape[1] * module.num_heads
            # Extra parameters can be exposed in TimmAttentionWrapper, this is a minimal example
            module_output = TimmAttentionWrapper(dim, module.num_heads, attn_mask=att_mask)
        for name, child in module.named_children():
            module_output.add_module(name, replace_attn_with_xformers_one(child, att_mask))
        del module
        return module_output

    # Now we can just patch our reference model, and get a sparse-aware variation
    model = replace_attn_with_xformers_one(model, mask)

Note that in practice exchanging all the attentions with a sparse alternative may not be a good idea, as the attentions closer to the output are not typically exhibiting a clear sparsity pattern. You can alter `replace_attn_with_xformers_one` above, or replace manually the attentions which would like to sparsify, but not all


.. _pytorch-image-models: https://github.com/rwightman/pytorch-image-models
.. _repository: https://github.com/facebookresearch/xformers




Create complex sparsity patterns
################################

This is best presented in an executable notebook, see [Creating complex sparsity patterns with xFormers](docs/source/2d_attention_patterns.ipynb). We can expose a couple of concepts here, though. Attention patterns are expressed over the attention matrix, and they intuitively represent the fact that an element in a sequence is allowed to "look at" another element in another sequence (same sequence in the self-attention case). With this framework, a local attention (an element is only able to look at its neighbors) is simply a diagonal pattern, while an element which can look at the whole sequence has a row+col matching mask. xFormers provide a couple of helpers to generate attention patterns, which can then be combined:

.. code-block:: python

    import xformers.components.attention.attention_patterns as AP

    # Couple of examples of attention patterns useful for vision
    H, W = 20, 30  # assuming a 20 x 30 sequence length

    # - axial attention
    axial_pattern = AP.axial_2d_pattern(H, W)

    # - local attention
    loc_2d_dist = AP.local_2d_pattern(H, W, distance=5, p=2.0)  # distance and thresholds are user defined

    # - random attention
    gaus_2d_dist = AP.local_2d_gausian_distribution(H, W, sigma=2)
    sparsity = 0.95
    num_non_zeros = int((H * W) ** 2 * (1 - sparsity))
    random_gaus_2d_pattern = AP.random_pattern_from_probability_matrix(gaus_2d_dist, num_non_zeros)

    # - combining different patterns
    combined_mask = axial_pattern | loc_2d_dist | random_gaus_2d_pattern


![Axial pattern](docs/assets/axial_pattern.png)
![Local pattern](docs/assets/local_pattern.png)
![Gaussian pattern](docs/assets/gaussian_pattern.png)
![Combined pattern](docs/assets/combined_pattern.png)
