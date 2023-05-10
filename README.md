

# DP-FedSAM

This repository contains the official implementation for these manuscripts:

> [Make Landscape Flatter in Differentially Private Federated Learning](https://arxiv.org/abs/2303.11242) (2023 CVPR)
>
> [Towards the Flatter Landscape and Better Generalization in Federated Learning under Client-level Differential Privacy](https://arxiv.org/abs/2305.00873v2) (Extension)

To defend the inference attacks and mitigate the sensitive information leakages in Federated Learning (FL), client-level Differentially Private FL (DPFL) is the de-facto standard for privacy protection by clipping local updates and adding random noise. However, existing DPFL methods tend to make a sharper loss landscape and have poorer weight perturbation robustness, resulting in severe performance degradation. To alleviate these issues, we propose a novel DPFL algorithm named DP-FedSAM, which leverages gradient perturbation to mitigate the negative impact of DP. Specifically, DP-FedSAM integrates Sharpness Aware Minimization (SAM) optimizer to generate local flatness models with better stability and weight perturbation robustness, which results in the small norm of local updates and robustness to DP noise, thereby improving the performance. From the theoretical perspective, we analyze in detail how DP-FedSAM mitigates the performance degradation induced by DP. Meanwhile, we give rigorous privacy guarantees with Rényi DP and present the sensitivity analysis of local updates. At last, we empirically confirm that our algorithm achieves state-of-the-art (SOTA) performance compared with existing SOTA baselines in DPFL. [1]



# Experiments

The implementation is provided in the folder `/fedml_api/dpfedsam`, while experiment is provided in the main file  `main_dpfedsam.py`.

### Platform

* Python: 3.9.7
* Pytorch: 1.12.0

### Model 
- a simple CNN model for EMNIST dataset
- ResNet-18 backbone for CIFAR-10 and CIFAR-100 datasets

### Dataset
- EMNIST

- CIFAR-10

- CIFAR-100

  

## Launch Experiments

~~~python
# EMNIST
python main_dpfedsam.py --model 'cnn_emnist' --dataset 'emnist' --partition_method 'dir' --partition_alpha 0.6 --batch_size 32 --lr 0.1 --lr_decay 0.998 --epochs 30  --client_num_in_total 500 --frac 0.1 --comm_round 200  --seed 2 --momentum 0.5 --C 0.2 --rho 0.5  --sigma 0.95 --gpu 0 --num_experiments 2

# CIFAR-10
python main_dpfedsam.py --model 'resnet18' --dataset 'cifar10' --partition_method 'dir' --partition_alpha 0.6 --batch_size 50 --lr 0.1 --lr_decay 0.998 --epochs 30  --client_num_in_total 500 --frac 0.1 --comm_round 300  --seed 2 --momentum 0.5 --C 0.2 --rho 0.5   --sigma 0.95 --gpu 3 --num_experiments 1

# CIFAR-100
python main_dpfedsam.py --model 'resnet18' --dataset 'cifar100' --partition_method 'dir' --partition_alpha 0.6 --batch_size 50 --lr 0.1 --lr_decay 0.998 --epochs 30  --client_num_in_total 500 --frac 0.1 --comm_round 300  --seed 2 --momentum 0.5 --C 0.2 --rho 0.5  --sigma 0.95 --num_experiments 2 --gpu 0
~~~

Explanations of arguments:

- `partition_method`: current supporting three types of data partition, one called 'dir' short for Dirichlet; one called 'n_cls' short for how many classes allocated for each client; and one called 'my_part' for partitioning all clients into PA shards with default latent Dir=0.3 distribution.
- `partition_alpha`: available parameters for data partition method.
- `client_num_in_total`: the number of clients
- `frac`: the selection fraction of total clients in each round.
- `comm_round`: how many round of communications we shoud use.
- `rho`: the perturbation radio for the SAM optimizer.
- `sigma`: the standard deviation of client-level DP noise.
- `C`: the threshold of clipping in DP.
- `num_experiments`: the number of experiments.



### Save Model
After finishing FL training process, the trained model file on CIFAR-10 dataset is saved in the `/save_model` folder in `.pth.tar` file format using the model with the `torch.save()` function. The saved model can be loaded and used to visualize the loss landscape and loss surface contour by using official code [loss-landscape](https://github.com/tomgoldstein/loss-landscape).

### Reference

[1] Shi, Y., Liu, Y., Wei, K., Shen, L., Wang, X., & Tao, D. (2023). Make Landscape Flatter in Differentially Private Federated Learning. *arXiv preprint arXiv:2303.11242*.

[2] Shi, Y., Wei, K., Shen, L., Liu, Y., Wang, X., Yuan, B., & Tao, D. (2023). Towards the Flatter Landscape and Better Generalization in Federated Learning under Client-level Differential Privacy. arXiv preprint arXiv:2305.00873.

### Citation
If you use this code, please cite the following reference:
```
@article{shi2023make,
  title={Make Landscape Flatter in Differentially Private Federated Learning},
  author={Shi, Yifan and Liu, Yingqi and Wei, Kang and Shen, Li and Wang, Xueqian and Tao, Dacheng},
  journal={arXiv preprint arXiv:2303.11242},
  year={2023}
}

@article{shi2023towards,
  title={Towards the Flatter Landscape and Better Generalization in Federated Learning under Client-level Differential Privacy},
  author={Shi, Yifan and Wei, Kang and Shen, Li and Liu, Yingqi and Wang, Xueqian and Yuan, Bo and Tao, Dacheng},
  journal={arXiv preprint arXiv:2305.00873},
  year={2023}
}
```
