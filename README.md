# [Dynamic Low-Rank Sparse Adaptation for Large Language Models](https://arxiv.org/abs/2502.14816)

Pytorch implementation of our paper accepted by ICLR 2025 -- **LoSA** (Dynamic **Lo**w-rank **S**parse **A**daptation).

## Abstract
Despite the efficacy of network sparsity in alleviating the deployment strain of Large Language Models (LLMs), it endures significant performance degradation. Applying Low-Rank Adaptation (LoRA) to fine-tune the sparse LLMs offers an intuitive approach to counter this predicament, while it holds shortcomings include: 1) The inability to integrate LoRA weights into sparse LLMs post-training, and 2) Insufficient performance recovery at high sparsity ratios. In this paper, we introduces dynamic **Lo**w-rank **S**parse **A**daptation (**LoSA**), a novel method that seamlessly integrates low-rank adaptation into LLM sparsity within a unified framework, thereby enhancing the performance of sparse LLMs without increasing the inference latency. In particular, LoSA dynamically sparsifies the LoRA outcomes based on the corresponding sparse weights during fine-tuning, thus guaranteeing that the LoRA module can be integrated into the sparse LLMs post-training. Besides, to achieve the optimal sparse model architecture, LoSA leverages Representation Mutual Information (RMI) as an indicator to determine the importance of layers, thereby dynamically determining the optimal layer-wise sparsity rates during fine-tuning. Predicated on this, LoSA adjusts the rank of the LoRA module based on the variability in layer-wise reconstruction errors, allocating an appropriate fine-tuning for each layer to reduce the output discrepancies between dense and sparse LLMs. Extensive experiments tell that LoSA can efficiently boost the efficacy of sparse LLMs within a few hours, without introducing any additional inferential burden. For example, LoSA reduced the perplexity of sparse LLaMA-2-7B by **68.73**$\downarrow$ and increased zero-shot accuracy by **16.32%**$\uparrow$, achieving a **2.60**$\times$ speedup on CPU and **2.23**$\times$ speedup on GPU, requiring only **45 minutes** of fine-tuning on **a single** NVIDIA A100 80GB GPU.


## Related Project

[A Simple and Effective Pruning Approach for Large Language Models](https://github.com/locuslab/wanda)

[SparseGPT: Massive Language Models Can be Accurately Pruned in One-Shot](https://github.com/ist-daslab/sparsegpt)

## Citation

if you find this repo is helpful, please cite our paper:
```
@inproceedings{
huang2025dynamic,
title={Dynamic Low-Rank Sparse Adaptation for Large Language Models},
author={Weizhong Huang and Yuxin Zhang and Xiawu Zheng and Liuyang and Jing Lin and Yiwu Yao and Rongrong Ji},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=oXh0939Zzq}
}
```
