# Spikingformer: A Key Foundation Model for Spiking Neural Networks, [AAAI 2026](https://arxiv.org/abs/2304.11954)

# Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network, [Arxiv 2023](https://arxiv.org/abs/2304.11954)
Spikingformer is a pure event-driven transformer-based spiking neural network (**75.85% top-1** accuracy on ImageNet-1K, **+ 1.04%** and **significantly reduces energy consumption by 57.34%** compared with Spikformer). To our best knowledge, this is the first time that **a pure event-driven transformer-based SNN** has been developed in 2023/04.


<p align="center">
<img src="https://github.com/zhouchenlin2096/Spikingformer/blob/master/imgs/Spikingformer-Architecture.png">
</p>

## News
[2025.11.8] Accepted by AAAI 2026.

[2024.2.23] Update energy_consumption_calculation of Spikingformer or Spikformer on ImageNet.

[2023.9.11] Update origin_logs and cifar10 trained model.

[2023.8.18] Update trained models.

## Reference
If you find this repo useful, please consider citing:
```
@article{zhou2023spikingformer,
  title={Spikingformer: Spike-driven Residual Learning for Transformer-based Spiking Neural Network},
  author={Zhou, Chenlin and Yu, Liutao and Zhou, Zhaokun and Zhang, Han and Ma, Zhengyu and Zhou, Huihui and Tian, Yonghong},
  journal={arXiv preprint arXiv:2304.11954},
  year={2023},
  url={https://arxiv.org/abs/2304.11954}
}

@article{zhou2024direct,
  title={Direct training high-performance deep spiking neural networks: a review of theories and methods},
  author={Zhou, Chenlin and Zhang, Han and Yu, Liutao and Ye, Yumin and Zhou, Zhaokun and Huang, Liwei and Ma, Zhengyu and Fan, Xiaopeng and Zhou, Huihui and Tian, Yonghong},
  journal={Frontiers in Neuroscience},
  volume={18},
  pages={1383844},
  year={2024},
  publisher={Frontiers Media SA}
}
```

## Main results on ImageNet-1K

| Model               | Resolution| T |  Param.     | FLOPs   |  Power |Top-1 Acc| Download |
| :---:               | :---:     | :---:  | :---:       |  :---:  |  :---:    |:---: |:---: |
| Spikingformer-8-384 | 224x224   | 4 |  16.81M     | 3.88G   | 4.69 mJ   |72.45  |   -    |
| Spikingformer-8-512 | 224x224   | 4 |  29.68M     | 6.52G  | 7.46 mJ   |74.79  |     -  |
| Spikingformer-8-768 | 224x224   | 4  |  66.34M     | 12.54G  | 13.68 mJ  |75.85  |   [here](https://pan.baidu.com/s/1LsECpFOxh30O3vHWow8OGQ) |

All download passwords: abcd

<!-- 
| Spikformer-8-384 | 224x224    |  16.81M     | 6.82G   | 12.43  mJ              |70.24  |
| Spikformer-8-512 | 224x224    |  29.68M     | 11.09G  | 18.82  mJ             |73.38  |
| Spikformer-8-768 | 224x224    |  66.34M     | 22.09G  | 32.07  mJ             |74.81  |
-->

## Main results on CIFAR10/CIFAR100

| Model                | T      |  Param.     | CIFAR10 Top-1 Acc| Download  |CIFAR100 Top-1 Acc|
| :---:                | :---:  | :---:       |  :---:  |:---:   |:---: |
| Spikingformer-4-256  | 4      |  4.15M     | 94.77   |   -   |77.43  |
| Spikingformer-2-384  | 4      |  5.76M     | 95.22   |   -   |78.34  |
| Spikingformer-4-384  | 4      |  9.32M     | 95.61    |   -  |79.09  |
| Spikingformer-4-384-400E  | 4      |  9.32M     | 95.81    | [here](https://pan.baidu.com/s/1mjpD2gtz5ZX0M8N3jobjzA ) |79.21  |

All download passwords: abcd

## Main results on CIFAR10-DVS/DVS128

| Model               | T      |  Param.     |  CIFAR10 DVS Top-1 Acc  | DVS 128 Top-1 Acc|
| :---:               | :---:  | :---:       | :---:                   |:---:            |
| Spikingformer-2-256 | 10     |  2.57M      | 79.9                    | 96.2            |
| Spikingformer-2-256 | 16     |  2.57M      | 81.3                    | 98.3            |


## Requirements
timm==0.6.12; cupy==11.4.0; torch==1.12.1; spikingjelly==0.0.0.0.12; pyyaml; 

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).
```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train
### Training  on ImageNet
Setting hyper-parameters in imagenet.yml

```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Testing ImageNet Val data
Download the trained model first [here](https://pan.baidu.com/s/1LsECpFOxh30O3vHWow8OGQ), passwords: abcd
```
cd imagenet
python test.py
```

### Training  on CIFAR10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```

### Training  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
cd cifar10
python train.py
```

### Training  on DVS128 Gesture
```
cd dvs128-gesture
python train.py
```

### Training  on CIFAR10-DVS
```
cd cifar10-dvs
python train.py
```

### Energy Consumption Calculation on ImageNet
Download the trained model first [here](https://pan.baidu.com/s/1LsECpFOxh30O3vHWow8OGQ), passwords: abcd
```
cd imagenet
python energy_consumption_calculation_on_imagenet.py
```

### MeZO Fine-tuning (zeroth-order optimization)
Fine-tunes the ImageNet-pretrained Spikingformer-8-768 on CIFAR-10 with MeZO instead of backpropagation, comparing four sources of perturbation randomness. Separate from the training scripts above — see [`finetune/README.md`](finetune/README.md) for hyper-parameters, experiment variants and recorded results.
```
cd finetune
bash run.sh
```

### Spike Sparsity of the QKV Layers (`sparsity/`)
Measures how sparse the attention path actually is at inference time: the **per-timestep spatial firing rate** of the QKV spiking outputs, i.e. the binary spike tensors produced by `block[i].attn.q_lif / k_lif / v_lif` (after `q_conv → BN → LIF`). Those tensors have shape `(T, B, C, N)` with the time dimension not collapsed, so a forward hook can report each timestep separately:

```
firing_rate[t] = (number of 1-spikes at timestep t) / (total elements at timestep t)
```
averaged over the whole validation set, giving `T = 4` numbers per layer.

One script per dataset — each loads the matching pretrained checkpoint and its own `model.py`, runs inference only (no training), prints a per-layer table plus q/k/v and global aggregates, and writes a CSV:

| Script | Model | Checkpoint | Output |
|---|---|---|---|
| `measure_sparsity_cifar10.py` | Spikingformer-4-384, 32×32, 4 blocks | `../checkpoint-405.pth.tar` (epoch 405, 95.81% top-1) | `sparsity_cifar10_qkv.csv`, `sparsity_cifar10.log` |
| `measure_sparsity_cifar100.py` | Spikingformer-4-384, 32×32, 4 blocks | `../checkpoint-cifar100.pth.tar` (epoch 370, 79.64% top-1) | `sparsity_cifar100_qkv.csv` |
| `measure_sparsity_imagenet.py` | Spikingformer-8-768, 224×224, 8 blocks | `../checkpoint-284.pth.tar` | `sparsity_imagenet_qkv.csv`, `sparsity_imagenet.log` |

```
cd sparsity
python measure_sparsity_cifar10.py
```

`T = 4` for all three; batch size is 64 for CIFAR and 32 for ImageNet. All three checkpoints are in the repo root, and every path is a repo-relative constant at the top of each script.

- The two **CIFAR** scripts run as-is — they read `../finetune/data`, which holds both CIFAR-10 and CIFAR-100.
- The **ImageNet** script additionally needs the ImageNet validation set at `../imagenet/data/val_blurred` (an `ImageFolder` layout). That data is not bundled here; point `VAL_DIR` at your own copy.

**Recorded results** (firing rate in %, averaged over all blocks):

| | t=0 | t=1 | t=2 | t=3 | mean |
|---|---:|---:|---:|---:|---:|
| **CIFAR-10** — q_lif | 10.65 | 12.73 | 11.66 | 11.62 | 11.66 |
| k_lif | 6.58 | 8.02 | 8.01 | 8.33 | 7.73 |
| v_lif | 4.93 | 7.62 | 7.55 | 7.84 | 6.99 |
| *all QKV* | 7.38 | 9.46 | 9.07 | 9.27 | **8.79** |
| **CIFAR-100** — q_lif | 12.85 | 12.50 | 11.96 | 11.88 | 12.30 |
| k_lif | 6.14 | 7.21 | 7.29 | 7.40 | 7.01 |
| v_lif | 5.82 | 7.76 | 7.68 | 7.78 | 7.26 |
| *all QKV* | 8.27 | 9.16 | 8.98 | 9.02 | **8.86** |
| **ImageNet** — q_lif | 7.59 | 8.45 | 8.54 | 8.16 | 8.19 |
| k_lif | 3.23 | 3.22 | 3.19 | 3.13 | 3.19 |
| v_lif | 5.42 | 6.02 | 6.16 | 5.83 | 5.86 |
| *all QKV* | 5.41 | 5.90 | 5.96 | 5.71 | **5.74** |

Fewer than 10% of the QKV activations spike on any timestep, and only 5.74% on ImageNet — the QKV path is highly sparse. `q_lif` fires 2–4× more often than `k_lif` consistently across all three datasets, and firing rates decrease with depth (per-block figures are in the CSVs). Timestep `t=0` is the sparsest, reflecting the LIF membrane potential still charging up.

## A Handwriting Error Correction in Manuscript
In neuromorphic datasets, the preprocessing (transforming events into frames) of neuromorphic datasets is according to SEW or SpikingJelly. The event stream comprises four dimensions: the event’s coordinate (x, y), time (t), and polarity (p). We split the event’s number N into T (the simulating time-step) slices with nearly the same number of events in each slice and integrate events into frames. It is a pity that Equation 20 in the manuscript is a formula mistake, we corrected it as follows:
$$E_{Spikingformer}^{neuro}=E_{A C} \times\left(\sum_{i=2}^N S O P_{{Conv} }^i+\sum_{j=1}^M S O P_{{SSA}}^j\right)+E_{M A C} \times\left(FLOP_{{Conv}}^1\right)$$


## Acknowledgement & Contact Information
Related project: [spikformer](https://github.com/ZK-Zhou/spikformer), [pytorch-image-models](https://github.com/huggingface/pytorch-image-models), [CML](https://github.com/zhouchenlin2096/Spikingformer-CML), [spikingjelly](https://github.com/fangwei123456/spikingjelly).

The code of Spikingformer<sup>†</sup>: [link](https://github.com/zhouchenlin2096/Spikingformer-CML)

Spikingformer-8-768 on Google Drive: [link](https://drive.google.com/drive/folders/1DE4qm-9vKNwDfJAdYYrDdPFc7_cGo2nK?usp=drive_link)

Spikingformer-4-384-400E on Google Drive: [link](https://drive.google.com/drive/folders/1LES30-AsGM02xkWftpUeh_WudhwDZzzW?usp=drive_link)

