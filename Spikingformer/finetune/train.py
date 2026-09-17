import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'imagenet'))

import argparse
import yaml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from functools import partial
from timm.models import create_model
from spikingjelly.clock_driven import functional
from mezo import MeZO

import model  # imagenet/model.py


# ─────────────────────────────────────────
# 1. 参数配置
# ─────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description='MeZO Fine-tuning on CIFAR10/100')
    parser.add_argument('-c', '--config', default='config.yml', type=str,
                        help='YAML 配置文件路径 (default: config.yml)')

    # 数据集
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--data_dir', default='./data', type=str)

    # 预训练权重
    parser.add_argument('--pretrained', default='../checkpoint-284.pth.tar', type=str)

    # 模型结构
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--embed_dims', default=768, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--depths', default=8, type=int)
    parser.add_argument('--T', default=4, type=int)

    # 冻结策略
    parser.add_argument('--freeze_ratio', default=0.75, type=float)

    # MeZO 超参数
    parser.add_argument('--random_mode', default='PGUST',
                        choices=['Randn', 'Randint', 'PGUXoR', 'PGUReuse'])
    parser.add_argument('--seed', default='0x123456789', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epsilon', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--perturb_nums', default=1, type=int)

    # 训练设置
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--output_dir', default='./output', type=str)

    # 先解析 config 文件路径
    args, remaining = parser.parse_known_args()

    # 读取 YAML，作为默认值
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    # 再完整解析（命令行参数可覆盖 YAML）
    args = parser.parse_args()
    return args


# ─────────────────────────────────────────
# 2. 数据集
# ─────────────────────────────────────────
def build_dataset(args):
    # 使用 ImageNet 归一化（与预训练模型一致）
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    interpolation = transforms.InterpolationMode.BICUBIC

    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(args.img_size),
        # transforms.RandomCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        #transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=interpolation),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    if args.dataset == 'cifar10':
        num_classes = 10
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=val_transform)
    else:  # cifar100
        num_classes = 100
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(
            root=args.data_dir, train=False, download=True, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader, num_classes


# ─────────────────────────────────────────
# 3. 模型构建 + 加载预训练权重
# ─────────────────────────────────────────
def build_model(args, num_classes):
    m = create_model(
        'Spikingformer',
        img_size_h=args.img_size, img_size_w=args.img_size,
        patch_size=16, embed_dims=args.embed_dims,
        num_heads=args.num_heads, mlp_ratios=4,
        in_channels=3, num_classes=num_classes, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=args.depths, sr_ratios=1, T=args.T
    )

    # 加载预训练权重，跳过 head（类别数不同）
    ckpt = torch.load(args.pretrained, map_location='cpu')
    sd = ckpt['state_dict']
    head_weight = sd.get('head.weight')
    if head_weight is not None and head_weight.shape[0] != num_classes:
        sd = {k: v for k, v in sd.items() if not k.startswith('head')}
        print(f'  [Info] head classes mismatch ({head_weight.shape[0]} vs {num_classes}), skipping head')
    missing, unexpected = m.load_state_dict(sd, strict=False)
    print(f'[Model] Pretrained weights loaded')
    print(f'  missing keys : {missing}')
    print(f'  unexpected   : {unexpected}')

    return m


# ─────────────────────────────────────────
# 4. 冻结参数
# ─────────────────────────────────────────
def freeze_params(model, freeze_ratio):
    """
    按层顺序冻结参数：
    冻结顺序为 patch_embed → block.0 → block.1 → ... → block.N → head
    freeze_ratio=0   : 全部参数可训练
    freeze_ratio=0.75: 冻结前75%的参数，后25%可训练（含head）
    freeze_ratio=1.0 : 只有head可训练
    """
    # 按顺序列出所有参数组（named_parameters保证顺序）
    all_params = [(n, p) for n, p in model.named_parameters()]
    total = len(all_params)
    freeze_count = int(total * freeze_ratio)

    for i, (name, param) in enumerate(all_params):
        if i < freeze_count:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'[冻结] freeze_ratio={freeze_ratio}')
    print(f'  可训练参数: {trainable:,}')
    print(f'  冻结参数  : {frozen:,}')


# ─────────────────────────────────────────
# 5. 验证
# ─────────────────────────────────────────
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = loss_fn(logits, target)
            functional.reset_net(model)

            total_loss += loss.item() * data.size(0)
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += data.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# ─────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────
def main():
    args = get_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # 数据集
    train_loader, val_loader, num_classes = build_dataset(args)
    print(f'[数据] 数据集: {args.dataset}, 类别数: {num_classes}')

    # 模型
    m = build_model(args, num_classes)
    m = m.to(device)

    # 冻结参数
    freeze_params(m, args.freeze_ratio)

    # loss 函数
    loss_fn = nn.CrossEntropyLoss()

    # MeZO 优化器
    optimizer = MeZO(
        device=device,
        model=m,
        random_mode=args.random_mode,
        seed=args.seed,
        perturb_nums=args.perturb_nums,
        lr=args.lr,
        epsilon=args.epsilon,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0
    best_epoch = 0
    best_state = None

    for epoch in range(args.epochs):
        m.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # MeZO 更新
            optimizer.step(data, target, loss_fn)
            functional.reset_net(m)

            # 只在打印时做一次额外推理
            if batch_idx % args.log_interval == 0:
                with torch.no_grad():
                    m.eval()
                    functional.reset_net(m)
                    logits = m(data)
                    loss = loss_fn(logits, target)
                    pred = logits.argmax(dim=1)
                    acc = pred.eq(target).sum().item() / data.size(0) * 100
                    m.train()
                print(f'Epoch [{epoch+1}/{args.epochs}] '
                      f'Batch [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f}  '
                      f'Acc: {acc:.2f}%')

        # 验证
        val_loss, val_acc = validate(m, val_loader, loss_fn, device)
        print(f'[Epoch {epoch+1}] Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.2f}%')

        # 记录最佳模型（只存内存，训练结束后统一写盘）
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            print(f'  → 新最佳: {best_acc:.2f}%  (epoch {best_epoch})')

    # 训练结束后统一保存最佳模型
    if best_state is not None:
        save_path = os.path.join(args.output_dir, f'{args.dataset}_best.pth')
        torch.save({
            'epoch': best_epoch,
            'state_dict': best_state,
            'val_acc': best_acc,
            'args': vars(args),
        }, save_path)
        print(f'\n[保存] 最佳模型已保存至 {save_path}')

    print(f'\n训练完成！最佳 Val Acc: {best_acc:.2f}%  (epoch {best_epoch})')


if __name__ == '__main__':
    main()
