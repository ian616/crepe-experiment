<div align="center">

# CREPE: Robust and Lightweight Pitch Estimation

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![arXiv](https://img.shields.io/badge/arXiv-1802.06182-B31B1B.svg)](https://arxiv.org/abs/1802.06182)

</div>

## Description

What it does

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

## References

> **CREPE: A Convolutional Representation for Pitch Estimation.**  
> Justin Salamon, Nicholas J. Bryan.  
> IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), 2018.  
> [arXiv:1802.06182](https://arxiv.org/abs/1802.06182)

> **MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.**  
> Andrew G. Howard et al.  
> arXiv preprint arXiv:1704.04861, 2017.  
> [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)
