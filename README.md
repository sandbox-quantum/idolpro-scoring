# Torchvina and TorchSA: Scoring functions used in IDOLpro - Guided Multi-objective Generative AI to Enhance Structure-based Drug Design

## Overview
This repo contains the scoring modules used in [Guided Multi-objective Generative AI to Enhance Structure-based Drug Design](https://arxiv.org/abs/2405.11785). Both of the modules in this repo are implemented with pytorch, and hence can be automatically differentiated to get gradients with respect to their inputs. The scoring modules present in this repo are:
* `torchvina`: A pytorch-based implementation of the Vina score.
* `torchSA`: An equivariant graph neural network ([PaiNN](https://proceedings.mlr.press/v139/schutt21a/schutt21a.pdf)) trained to predict the SA score of molecules.

## Installing required packages
To install all of the packages required to run the modules in this repo, run:
```bash
micromamba env create -f environment.yml
```
The environment assumes that it is being installed in a location with a GPU, and is currently configured to install `pytorch` for CUDA 12.1. To change the installation to the appropriate CUDA version, simply modify the following line in `environment.yml`: `- pytorch::pytorch-cuda=12.1`.

## Using torchvina and torchSA
Both of the scoring modules are made to take in RDKit molecules and return their respective scores. For an example of how to use torchvina, see [torchvina example](torchvina_example.ipynb). For an example of to use torchSA, see [torchSA example](torchsa_example.ipynb).
