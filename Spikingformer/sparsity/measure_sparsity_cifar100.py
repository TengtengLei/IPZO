"""
测算 Spikingformer (CIFAR-100) QKV 线性层输出的每时间步空间脉冲发放率。

目标层：block[i].attn.q_lif / k_lif / v_lif 的输出
  shape: (T, B, C, N)，T 未被合并
  firing_rate[t] = 时间步 t 中输出脉冲为 1 的个数 / 输出总元素数

运行方式：
    cd sparsity
    CUDA_VISIBLE_DEVICES=0 python measure_sparsity_cifar100.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cifar100'))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from functools import partial
from timm.models import create_model
from spikingjelly.clock_driven import functional

import model  # cifar100/model.py

# ── 配置 ──────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'checkpoint-cifar100.pth.tar')
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'finetune', 'data')
DEVICE     = 'cuda:0'
BATCH_SIZE = 64
T_STEPS    = 4
# ─────────────────────────────────────────────────────────────────


def build_model(checkpoint_path, device):
    net = create_model(
        'Spikingformer',
        pretrained=False,
        drop_rate=0., drop_path_rate=0.2, drop_block_rate=None,
        img_size_h=32, img_size_w=32,
        patch_size=4, embed_dims=384, num_heads=8, mlp_ratios=4,
        in_channels=3, num_classes=100, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=4, sr_ratios=1, T=4,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    net.load_state_dict(ckpt['state_dict'])
    print(f'[模型] 加载完成 | epoch={ckpt["epoch"]} | metric={ckpt.get("metric", "N/A")}')
    return net.to(device).eval()


def build_val_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std =[0.2470, 0.2435, 0.2616]),
    ])
    dataset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=False, transform=transform)
    print(f'[数据] 共 {len(dataset)} 张图')
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)


def measure_sparsity(net, val_loader, device, T=4):
    target_layers = {}
    for i in range(4):
        b = net.block[i]
        target_layers[f'block[{i}].attn.q_lif'] = b.attn.q_lif
        target_layers[f'block[{i}].attn.k_lif'] = b.attn.k_lif
        target_layers[f'block[{i}].attn.v_lif'] = b.attn.v_lif

    stats = {
        name: {'nonzero': [0] * T, 'total': [0] * T}
        for name in target_layers
    }
    handles = []

    for name, module in target_layers.items():
        def make_hook(n):
            def hook(mod, inp, out):
                spikes = out.detach().float()  # (T, B, C, N)
                for t in range(spikes.shape[0]):
                    xt = spikes[t]
                    stats[n]['nonzero'][t] += (xt != 0).sum().item()
                    stats[n]['total'][t]   += xt.numel()
            return hook
        handles.append(module.register_forward_hook(make_hook(name)))

    print('\n[推理] 在 CIFAR-100 val set 上测算 QKV 输出脉冲发放率...')
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(val_loader):
            images = images.to(device)
            _ = net(images)
            functional.reset_net(net)
            if (batch_idx + 1) % 20 == 0:
                print(f'  batch {batch_idx + 1}/{len(val_loader)}')

    for h in handles:
        h.remove()

    return {
        name: [s['nonzero'][t] / s['total'][t] for t in range(T)]
        for name, s in stats.items()
    }


def print_results(results, T=4):
    t_header = ''.join(f'    t={t}  ' for t in range(T))
    print(f'\n{"层名":<35}{t_header}  {"均值":>6}')
    print('-' * (35 + T * 10 + 8))

    for name, rates in results.items():
        avg = sum(rates) / T
        rates_str = ''.join(f'{r*100:>8.2f}%' for r in rates)
        print(f'{name:<35}{rates_str}  {avg*100:>6.2f}%')

    print()
    for qkv in ['q_lif', 'k_lif', 'v_lif']:
        rs = [r for n, r in results.items() if qkv in n]
        per_t = [sum(r[t] for r in rs) / len(rs) for t in range(T)]
        avg   = sum(per_t) / T
        rates_str = ''.join(f'{v*100:>8.2f}%' for v in per_t)
        print(f'[{qkv} 各block均值]{"":<13}{rates_str}  {avg*100:>6.2f}%')

    print()
    all_rates = list(results.values())
    overall_t = [sum(r[t] for r in all_rates) / len(all_rates) for t in range(T)]
    overall   = sum(overall_t) / T
    rates_str = ''.join(f'{v*100:>8.2f}%' for v in overall_t)
    print(f'[QKV 全局均值]{"":>21}{rates_str}  {overall*100:>6.2f}%')


def save_csv(results, save_path, T=4):
    import csv
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['layer'] + [f't={t}' for t in range(T)] + ['mean'])
        for name, rates in results.items():
            avg = sum(rates) / T
            writer.writerow([name] + [f'{r*100:.4f}' for r in rates] + [f'{avg*100:.4f}'])
    print(f'\n[保存] 结果已保存至 {save_path}')


def main():
    device     = torch.device(DEVICE)
    net        = build_model(CHECKPOINT, device)
    val_loader = build_val_loader(DATA_DIR, BATCH_SIZE)
    results    = measure_sparsity(net, val_loader, device, T=T_STEPS)
    print_results(results, T=T_STEPS)

    save_path = os.path.join(os.path.dirname(__file__), 'sparsity_cifar100_qkv.csv')
    save_csv(results, save_path, T=T_STEPS)


if __name__ == '__main__':
    main()
