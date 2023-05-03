This repo is based on the work of ([Weiss et al. 2021](https://arxiv.org/abs/2106.06981)) who developed the RASP programming language and (https://arxiv.org/abs/2301.05062) who developed a compiled for converting the RASP algorothms into the weights of a transformer model. 

We build on this work by extracting the compiled weights and loading them into a pytorch model, then freezing subsets of the layers and re-training the model to see if we can recover the compiled weights (or close to).

Note this project is a WIP.

Directory structure:

* `tracr_utils` contains the implementation of RASP and the compiler provided by (https://arxiv.org/abs/2301.05062)
* `model` contains a pytorch transformer model with the same architeture as is returned by the compiler
* `inference` contains load_weights.py which compiles RASP alogorithms then loads the weights of the JAX Haiku model returned into our pytorch implementation. We define an number of RASP algorithms in algorithms.py that can be compiled into models. Examples of running inference on this model are given in test/test_inference.
* `train` contains code for training partially frozen models on some of the algorithmic tasks defined in algorithms.py
