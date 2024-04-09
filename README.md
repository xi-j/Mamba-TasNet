# Mamba-TasNet

An official implementation of dual-path Mamba speech separation model.

arxiv: https://arxiv.org/abs/2403.18257

## Architecture

![architecture](figures/dpmamba.png)

## Training 

## Inference

See inference.ipynb. Model checkpoints are provided in the results folder.

## Performance
We slightly improved the performance from the paper. DPMamba (L) was trained for 200 instead of 150 epochs.
<img src="figures/performance.png" alt="performance" width="60%">

## Acknowledgement

We acknowledge the wonderful work of [Mamba](https://arxiv.org/abs/2312.00752) and [Vision Mamba](https://arxiv.org/abs/2401.09417). We borrowed their implementation of [Mamba](https://github.com/state-spaces/mamba) and [bidirectional Mamba](https://github.com/hustvl/Vim). The training recipes are adapted from [SpeechBrain](https://speechbrain.github.io).

## Citation
If you find this work useful, consider citing
```bibtex
@article{jiang2024dual,
  title={Dual-path Mamba: Short and Long-term Bidirectional Selective Structured State Space Models for Speech Separation},
  author={Jiang, Xilin and Han, Cong and Mesgarani, Nima},
  journal={arXiv preprint arXiv:2403.18257},
  year={2024}
}
```
