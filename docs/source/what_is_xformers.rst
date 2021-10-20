What is xFormers?
==================

Flexible Transformers, defined by interoperable and optimized building blocks.

.. image:: _static/logo.png
    :width: 700px
    :height: 205px
    :align: center


xFormers is focused on the following values

- **Field agnostic**. This library is not focused on any given field, by design.

- **Composable**. Ideally, break all the Transformer inspired models into a *block zoo*, which allows you to compose reference models but also study ablations or architecture search.

- **Extensible**. xFormers aims at being *easy to extend locally*, so that one can focus on a specific improvement, and easily compare it against the state of the art.

- **Optimized**. Reusing building blocks across domains means that engineering efforts can be more valued. And since you cannot improve what you cannot measure, xFormers is benchmark-heavy.

- **Tested**. Each and every of the variant in the repo is tested, alone and composed with the other relevant blocks. This happens automatically anytime somebody proposes a new variant through a PR.

- **Crowd Sourced**. PRs are really welcome, the state of the art is moving too fast for anything but a crowd sourced effort.


Installation
============

To install xFormers, it is recommended to use a dedicated virtual environment, as often with python, through `python-virtualenv` or `conda` for instance.
There are two ways you can install it:

Directly from the pip package
-----------------------------

You can also fetch the latest release from PyPi. This will not contain the wheels for the sparse attention kernels, for which you will need to build from source.

.. code-block:: bash

    conda create --name xformer_env
    conda activate xformer_env
    pip install xformers

Build from source (dev mode)
----------------------------

These commands will fetch the latest version of the code, create a dedicated `conda` environment, activate it then install xFormers from source. If you want to build the sparse attention CUDA kernels, please make sure that the next point is covered prior to running these instructions.

.. code-block:: bash

  git clone git@github.com:fairinternal/xformers.git
  conda create --name xformer_env python=3.8
  conda activate xformer_env
  cd xformers
  pip install -r requirements.txt
  pip install -e .


Sparse attention kernels
************************

Installing the CUDA-based sparse attention kernels (derived from Sputnik_ and GE-SpMM_) may require extra care, as this mobilizes the CUDA toolchain.
As a reminder, these kernels are built when you run `pip install -e .` and the CUDA buildchain is available (NVCC compiler).
Re-building can for instance be done via `python3 setup.py clean && python3 setup.py develop`,
or similarly wiping the `build` folder and redoing a `pip install -e .`

Some advices related to building these CUDA-specific components, tentatively adressing common pitfalls. Please make sure that:

* NVCC and the current CUDA runtime match. You can often change the CUDA runtime with `module unload cuda module load cuda/xx.x`, possibly also `nvcc`
* the version of GCC that you're using matches the current NVCC capabilities
* the `TORCH_CUDA_ARCH_LIST` env variable is set to the architures that you want to support. A suggested setup (slow to build but comprehensive) is `export TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.2;8.0;8.6"`

.. _Sputnik: https://github.com/google-research/sputnik
.. _GE-SpMM: https://github.com/hgyhungry/ge-spmm

Triton
******

Some parts of xFormers use [Triton](http://www.triton-lang.org), and will only expose themselves if Triton is installed, and a compatible GPU is present (nVidia GPU with tensor cores). If Triton was not installed as part of the testing procedure, you can install it directly by running `pip install triton`. You can optionally test that the installation is successful by running one of the Triton-related benchmarks, for instance `python3 xformers/benchmarks/benchmnark_triton_softmax.py`

Triton will cache the compiled kernels to `/tmp/triton` by default. If this becomes an issue, this path can be specified through the `TRITON_CACHE_DIR` environment variable.

Testing the installation
------------------------

This will run a benchmark of the attention mechanisms exposed by xFormers, and generate a runtime and memory plot.
If this concludes without errors, the installation is successful. This step is optional, and you will need some extra dependencies for it to
be able to go through : `pip install -r requirements-benchmark.txt`.

Once this is done, you can run this particular benchmark as follows:

.. code-block:: bash

    python3 xformers/benchmarks/benchmark_encoder.py --activations relu  --plot -emb 256 -bs 32 -heads 16
