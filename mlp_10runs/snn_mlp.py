
import math
import galois
import pickle
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from spikingjelly.activation_based import neuron, layer, functional, surrogate
from tqdm import tqdm  # progress bar
import random
import numpy as np
import os

# Device configuration (auto-select GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 1. Poisson encoder
class PoissonEncoder(nn.Module):
    def __init__(self, time_step, seed):
        super().__init__()
        self.time_step = time_step  # number of time steps
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    def forward(self, x):
        # x: [B, C, H, W]
        # expand the time dimension
        x = x.unsqueeze(0).repeat(self.time_step, 1, 1, 1, 1)  # [T, B, C, H, W]
        random_tensor = torch.rand(x.size(), device=x.device, generator=self.generator)

        # generate Poisson spikes
        return (random_tensor < x).float()  # bool tensor -> float


# 2. SNN architecture
class SNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, time_step, seed):
        super().__init__()
        self.out_features = out_features
        self.time_step = time_step

        # Network
        self.fc1 = layer.Linear(in_features, hidden_features)
        self.lif1 = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            surrogate_function=surrogate.Sigmoid()
        )

        self.fc2 = layer.Linear(hidden_features, out_features)
        self.lif2 = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            surrogate_function=surrogate.Sigmoid()
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                gen = torch.Generator()
                gen.manual_seed(seed)
                # Optionally initialize as int8 weights first
                # int8_weights = torch.randint(-128, 128, m.weight.shape, dtype=torch.int8, generator=gen)
                # Convert int8 to float32 and scale
                # m.weight.data = int8_weights.float() / 128.0
                # nn.init.constant_(m.bias, 0)
                nn.init.trunc_normal_(m.weight, std=0.02, generator=gen)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input expected as [T, B, C, H, W]
        batch_size = x.shape[1]
        outputs = torch.zeros((self.time_step, batch_size, self.out_features), device=x.device)

        for t in range(self.time_step):
            x_t = x[t].view(batch_size, -1)  # flatten
            x_t = self.fc1(x_t)
            x_t = self.lif1(x_t)

            x_t = self.fc2(x_t)
            x_t = self.lif2(x_t)
            outputs[t] = x_t

        # return the mean output over all time steps
        return outputs.mean(0)


class PGUXoR:
    def __init__(self, dim, seed='0x123456789', num_rows=8, num_cols=16, num_quo_lfsr=16, bit_width=8, device=None):
        self.dim = dim
        self.device = device
        self.root = './poly/'
        os.makedirs(self.root, exist_ok=True)

        if bit_width == 1:
            segments_per_lfsr = 36
            stride = 1
        elif bit_width == 4:
            segments_per_lfsr = 9  # 4*9=36, stride=4
            stride = 4
        elif bit_width == 8:
            segments_per_lfsr = 8  # 4*7+8=36, stride=4
            stride = 4
        elif bit_width == 16:
            segments_per_lfsr = 6  # 4*5+16=36, stride=4
            stride = 4
        else:
            raise ValueError("bit_width must be 1,4,8 or 16")

        # Number of LFSRs
        self.num_rows = num_rows
        self.num_quo_lfsr = num_quo_lfsr
        self.num_cols = num_cols
        self.num_arrays = math.ceil(dim / (segments_per_lfsr * num_rows * num_cols * num_quo_lfsr))
        self.lfsr_len = 36

        # Initialize seed and polynomials
        self.seed = int(seed, 16)
        self.primitives = self._load_or_generate_primitives()
        self._init_lfsr()
        self._reseed()

        # Segment start indices
        self.starts = torch.arange((segments_per_lfsr - 1) * stride, -1, -stride, device=device)
        self.mask_w = (1 << bit_width) - 1

        self.output_dtype = torch.int16 if bit_width == 16 else torch.int8
        self.output_buffer = torch.empty(dim, dtype=self.output_dtype, device=device)

    # Load 10000 primitive polynomials of degree 36
    def _load_or_generate_primitives(self):
        cache_file = os.path.join(self.root, 'primitives_degree36_10k.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)  # load all
        else:
            print("Generating 10000 polynomials...")
            poly_gen = galois.primitive_polys(2, degree=self.lfsr_len)
            primitives = [next(poly_gen) for _ in range(10000)]  # pre-generate enough
            with open(cache_file, 'wb') as f:
                pickle.dump(primitives, f)
            return primitives  # return all

    def _init_lfsr(self):
        # Initialize state tensors (stored as integers)
        # [num_arrays]
        self.seed_states = torch.full(
            (self.num_arrays,), self.seed,
            dtype=torch.int64, device=self.device
        )
        # [num_arrays, num_quo_lfsr]
        self.quo_states = torch.full(
            (self.num_arrays, self.num_quo_lfsr), self.seed,
            dtype=torch.int64, device=self.device
        )
        # [num_arrays, num_rows, num_cols]
        self.rem_states = torch.full(
            (self.num_arrays, self.num_rows, self.num_cols), self.seed,
            dtype=torch.int64, device=self.device
        )

        # Polynomial mask
        def build_mask(poly):
            poly_int = int(poly) >> 1  # drop the top bit; convert galois polynomial to int
            return poly_int

        # Quotient LFSR masks [num_quo_lfsr]
        self.quo_mask = torch.tensor([
            build_mask(p) for p in self.primitives[0:self.num_quo_lfsr]
        ], dtype=torch.int64, device=self.device)

        # Expand masks per array [num_arrays, num_quo_lfsr]
        self.quo_masks = self.quo_mask.unsqueeze(0).expand(self.num_arrays, self.num_quo_lfsr)

        # Remainder LFSR masks [num_rows, num_cols]
        rem_polys = self.primitives[self.num_quo_lfsr:self.num_quo_lfsr + self.num_rows * self.num_cols]
        self.rem_mask = torch.zeros(
            (self.num_rows, self.num_cols),
            dtype=torch.int64,
            device=self.device
        )
        # Fill polynomial masks in column-major order
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                idx = col * self.num_rows + row
                self.rem_mask[row, col] = build_mask(rem_polys[idx])

        # Expand masks per array [num_arrays, num_rows, num_cols]
        self.rem_masks = self.rem_mask.unsqueeze(0).expand(
            self.num_arrays, self.num_rows, self.num_cols)

        # Seed LFSR masks [num_arrays]
        seed_polys = self.primitives[self.num_quo_lfsr + self.num_rows * self.num_cols:]
        self.seed_masks = torch.tensor(
            [build_mask(p) for p in seed_polys[:self.num_arrays]],
            dtype=torch.int64,
            device=self.device
        )

    @staticmethod
    def _galois_lfsr_step(states, masks):
        feedback = states & 1  # get the lowest bit
        shifted = states >> 1  # int64 logical right shift; bit 37 is 0 so bit 36 becomes 0, then apply feedback and mask to set the top bit
        return shifted ^ (feedback * masks)

    def _update(self):
        self.quo_states = self._galois_lfsr_step(self.quo_states, self.quo_masks)
        self.rem_states = self._galois_lfsr_step(self.rem_states, self.rem_masks)

    def _reseed(self):
        for _ in range(400):
            self.seed_states = self._galois_lfsr_step(self.seed_states, self.seed_masks)

        # seed_states [num_arrays]
        self.quo_states[:] = self.seed_states.unsqueeze(1).expand(self.num_arrays, self.num_quo_lfsr)
        self.rem_states[:] = self.seed_states.unsqueeze(1).unsqueeze(2).expand(
            self.num_arrays, self.num_rows, self.num_cols)

    def _xor_segment(self):
        # quo_states: [num_arrays, num_quo_lfsr]
        # rem_states: [num_arrays, num_rows, num_cols]
        # xor_result: [num_arrays, num_quo_lfsr, num_rows, num_cols]

        xor_result = self.quo_states[:, :, None, None] ^ self.rem_states[:, None, :, :]
        segments = xor_result.unsqueeze(-1) >> self.starts & self.mask_w

        # reshape to [num_arrays, per_array]
        segments_reshaped = segments.reshape(self.num_arrays, -1)
        return segments_reshaped.reshape(-1)[:self.dim].to(self.output_dtype)

    def step(self):
        self._update()
        random = self._xor_segment()
        output = self.output_buffer[:random.shape[0]].copy_(random)
        return output


class PGUReuse:
    def __init__(self, dim, seed='0x123456789', num_rows=8, num_cols=16, num_quo_lfsr=16, bit_width=8, device=None):
        self.dim = dim
        self.device = device
        self.root = './poly/'
        os.makedirs(self.root, exist_ok=True)

        if bit_width == 1:
            segments_per_lfsr = 36
            stride = 1
        elif bit_width == 4:
            segments_per_lfsr = 9  # 4*9=36, stride=4
            stride = 4
        elif bit_width == 8:
            segments_per_lfsr = 8  # 4*7+8=36, stride=4
            stride = 4
        elif bit_width == 16:
            segments_per_lfsr = 6  # 4*5+16=36, stride=4
            stride = 4
        else:
            raise ValueError("bit_width must be 1,4,8 or 16")

        # Number of LFSRs
        self.num_rows = num_rows
        self.num_quo_lfsr = num_quo_lfsr
        self.num_cols = num_cols
        self.num_arrays = math.ceil(dim / (segments_per_lfsr * num_rows * num_quo_lfsr * num_cols))
        self.lfsr_len = 36

        # Initialize seed and polynomials
        self.seed = int(seed, 16)
        self.primitives = self._load_or_generate_primitives()
        self._init_lfsr()
        self._reseed()

        seq_row = torch.arange(num_rows)
        shift_row = torch.arange(num_quo_lfsr)
        index = (seq_row.unsqueeze(0) - shift_row.unsqueeze(1)) % num_rows
        flat_index = index.flatten()
        self.index = flat_index
        #self.index = torch.cat([flat_index, torch.tensor([0])])
        self.shift_amount = 0

        # Segment start indices
        self.starts = torch.arange((segments_per_lfsr-1) * stride, -1, -stride, device=device)
        self.mask_w = (1 << bit_width) - 1

        self.output_dtype = torch.int16 if bit_width == 16 else torch.int8
        self.output_buffer = torch.empty(dim, dtype=self.output_dtype, device=device)

    # Load 10000 primitive polynomials of degree 36
    def _load_or_generate_primitives(self):
        cache_file = os.path.join(self.root, 'primitives_degree36_10k.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)  # load all
        else:
            print("Generating 10000 polynomials...")
            poly_gen = galois.primitive_polys(2, degree=self.lfsr_len)
            primitives = [next(poly_gen) for _ in range(10000)]  # pre-generate enough
            with open(cache_file, 'wb') as f:
                pickle.dump(primitives, f)
            return primitives  # return all

    def _init_lfsr(self):
        # Initialize state tensors (stored as integers)
        # [num_arrays]
        self.seed_states = torch.full(
            (self.num_arrays,), self.seed,
            dtype=torch.int64, device=self.device
        )
        # [num_arrays, num_rows, num_cols]
        self.rem_states = torch.full(
            (self.num_arrays, self.num_rows, self.num_cols), self.seed,
            dtype=torch.int64, device=self.device
        )

        # Polynomial mask
        def build_mask(poly):
            poly_int = int(poly) >> 1  # drop the top bit; convert galois polynomial to int
            return poly_int

        # Remainder LFSR masks [num_rows, num_cols]
        rem_polys = self.primitives[0:self.num_rows*self.num_cols]
        self.rem_mask = torch.zeros(
            (self.num_rows, self.num_cols),
            dtype=torch.int64,
            device=self.device
        )
        # Fill polynomial masks in column-major order
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                idx = col * self.num_rows + row
                self.rem_mask[row, col] = build_mask(rem_polys[idx])

        # Expand masks per array [num_arrays, num_rows, num_cols]
        self.rem_masks = self.rem_mask.unsqueeze(0).expand(
            self.num_arrays, self.num_rows, self.num_cols)

        # Seed LFSR masks [num_arrays]
        seed_polys = self.primitives[self.num_rows * self.num_cols:]
        self.seed_masks = torch.tensor(
            [build_mask(p) for p in seed_polys[:self.num_arrays]],
            dtype=torch.int64,
            device=self.device
        )

    @staticmethod
    def _galois_lfsr_step(states, masks):
        feedback = states & 1  # get the lowest bit
        shifted = states >> 1  # int64 logical right shift; bit 37 is 0 so bit 36 becomes 0, then apply feedback and mask to set the top bit
        return shifted ^ (feedback * masks)

    def _update(self):
        self.rem_states = self._galois_lfsr_step(self.rem_states, self.rem_masks)

    def _reseed(self):
        for _ in range(400):
            self.seed_states = self._galois_lfsr_step(self.seed_states, self.seed_masks)

        # seed_states [num_arrays]
        self.rem_states[:] = self.seed_states.unsqueeze(1).unsqueeze(2).expand(
            self.num_arrays, self.num_rows, self.num_cols)

    def _shift_segment(self):
        # quo_states: [num_arrays, num_quo_lfsr]
        # rem_states: [num_arrays, num_rows, num_cols]
        # xor_result: [num_arrays, num_quo_lfsr, num_rows, num_cols]
        # rem_states_expand = self.rem_states.unsqueeze(1).expand(-1, self.num_quo_lfsr, -1, -1)

        self.shift_index = torch.roll(self.index, self.shift_amount)
        rem_states_expand = self.rem_states[:, self.shift_index, :]
        segments = rem_states_expand.unsqueeze(-1) >> self.starts & self.mask_w

        # reshape to [num_arrays, per_array]
        # segments_reshaped = segments.reshape(self.num_arrays, -1)
        return segments.reshape(-1)[:self.dim].to(self.output_dtype)

    def step(self):
        self._update()
        self.shift_amount = -((self.shift_amount + 1) % (self.num_quo_lfsr * self.num_rows))
        random = self._shift_segment()
        output = self.output_buffer[:random.shape[0]].copy_(random)
        return output


# 3. MeZO optimizer
class MeZO:
    def __init__(self, model, random_mode='PGUReuse', lr=0.01, mu=0.01, T_max=30, eta_min=0.1):
        self.model = model
        self.random_mode = random_mode
        self.lr = lr
        self.base_lr = lr  # keep the initial learning rate
        self.mu = mu
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.dim = sum(p.numel() for p in self.params)  # precompute total parameter count
        self.T_max = T_max
        self.eta_min = eta_min
        self.epoch = 0  # track current epoch

        # Select the perturbation generator by random source; PGU uses a fixed seed
        # 0x123456789 (consistent with existing results); run-to-run differences come
        # from weight initialization and data shuffling order, not from the perturbation stream.
        if random_mode == 'PGUReuse':
            self.pgu = PGUReuse(self.dim, bit_width=8, device=device)
        elif random_mode == 'PGUXoR':
            self.pgu = PGUXoR(self.dim, bit_width=8, device=device)
        else:  # Randn / Randint use torch native RNG, no PGU needed
            self.pgu = None

    def step(self, loss_fn, data, target):
        with torch.no_grad():
            # 1. Generate the perturbation vector u; all four sources are normalized to variance 1 for fair comparison
            if self.random_mode == 'Randn':
                # normally distributed random numbers
                u = torch.normal(mean=0, std=1, size=(self.dim,), device=device)
            elif self.random_mode == 'Randint':
                # torch native uniformly distributed int8 random numbers
                u_int8 = torch.randint(-128, 128, (self.dim,), dtype=torch.int8, device=device)
                u = u_int8.float() / 256.0  # map to [-0.5, 0.5] (127.5/255 ~ 0.5)
                u = u * (12 ** 0.5)          # sqrt(12) ~ 3.464, adjusts variance to 1
            else:
                # uniformly distributed int8 random numbers from PGU (PGUXoR / PGUReuse)
                u_int8 = self.pgu.step()
                u = u_int8.float() / 256.0
                u = u * (12 ** 0.5)

            # 2. Positive perturbation evaluation
            functional.reset_net(self.model)
            self._perturb_params(u, +self.mu)
            y_pos = loss_fn(self.model(data), target).item()

            # 3. Negative perturbation evaluation
            functional.reset_net(self.model)
            self._perturb_params(u, -2 * self.mu)
            y_neg = loss_fn(self.model(data), target).item()

            # 4. Restore original parameters
            self._perturb_params(u, +self.mu)

            # 5. Gradient estimation and update
            grad_est = (y_pos - y_neg) / (2 * self.mu) * u
            self._update_params(-self.lr * grad_est)

            # 6. Optionally quantize/dequantize all parameters after update
            # self._quantize_and_dequantize_params()

    def update_lr(self):
        self.epoch += 1
        if self.epoch <= self.T_max:
            # cosine annealing formula
            self.lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
                      (1 + math.cos(math.pi * self.epoch / self.T_max))
        else:
            self.lr = self.eta_min

    def get_lr(self):
        return self.lr

    def _perturb_params(self, u, scale):
        idx = 0
        for p in self.params:
            numel = p.numel()
            delta = scale * u[idx:idx + numel].view_as(p.data)
            p.data.add_(delta)
            idx += numel

    def _update_params(self, update):
        idx = 0
        for p in self.params:
            numel = p.numel()
            p.data.add_(update[idx:idx + numel].view_as(p.data))
            idx += numel

    def _quantize_and_dequantize_params(self, bit_width=8):
        # Set quantization parameters by bit width
        if bit_width == 8:
            dtype = torch.int8
            max_val = 128
            clamp_min, clamp_max = -128, 127
        elif bit_width == 16:
            dtype = torch.int16
            max_val = 32768
            clamp_min, clamp_max = -32768, 32767
        else:
            raise ValueError("bit_width must be 8 or 16")

        for p in self.params:
            # Find the max absolute value to determine the scale factor
            p_max = torch.max(torch.abs(p.data))
            if p_max == 0:
                # avoid division by zero
                quantized = torch.zeros_like(p.data, dtype=dtype)
            else:
                # scale to [-1, 1] and quantize
                scaled = p.data / p_max
                quantized = (scaled * max_val).round().clamp(clamp_min, clamp_max).to(dtype)

            # dequantize: convert quantized parameters back to float
            p.data = quantized.float() / max_val * p_max


# 4. Data loading and preprocessing
def get_data_loaders(batch_size, test_batch_size, seed):
    transform = transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)

    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    # On Windows it is recommended to set num_workers=0
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


# 5. Training and evaluation
def train(model, device, train_loader, optimizer, epoch, encoder):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total_samples = 0  # for accuracy computation

    with tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            data = encoder(data)

            optimizer.step(loss_fn, data, target)

            with torch.no_grad():
                functional.reset_net(model)
                output = model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item() * data.size(1)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total_samples += data.size(1)
                pbar.set_postfix({'Loss': loss.item(), 'Acc': 100. * correct / total_samples})

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f"Train Epoch: {epoch} | "
          f"Learning rate: {optimizer.get_lr():.4f} | "
          f"Loss: {avg_loss:.4f} | "
          f"Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


# 6. Test function
def test(model, device, test_loader, encoder):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    total_samples = 0  # for accuracy computation

    with torch.no_grad(), tqdm(test_loader, desc="Testing", unit="batch") as pbar:
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            data = encoder(data)
            functional.reset_net(model)
            output = model(data)
            test_loss += loss_fn(output, target).item() * data.size(1)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += data.size(1)
            pbar.set_postfix({'Acc': 100. * correct / total_samples})

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
    return test_loss, accuracy


# 7. Main program
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reconfigurable SNN-MNIST Training')
    parser.add_argument('--in_features', type=int, default=1024, help='input dimension')
    parser.add_argument('--hidden_features', type=int, default=128, help='hidden dimension')
    parser.add_argument('--out_features', type=int, default=10, help='out dimension')
    parser.add_argument('--time_step', type=int, default=16, help='number of time steps for SNN')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='training epochs')
    parser.add_argument('--lr', type=float, default=2.5, help='learning rate')
    parser.add_argument('--eta_min', type=float, default=0.01, help='minimum learning rate')
    parser.add_argument('--mu', type=float, default=1.0, help='perturbation scaling factor')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='model saving path')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--random_mode', type=str, default='PGUReuse',
                        choices=['Randn', 'Randint', 'PGUXoR', 'PGUReuse'],
                        help='random source for MeZO perturbation')
    args = parser.parse_args()

    set_seed(seed=args.seed)
    encoder = PoissonEncoder(time_step=args.time_step, seed=args.seed)
    model = SNN(in_features=args.in_features, hidden_features=args.hidden_features,
                out_features=args.out_features, time_step=args.time_step, seed=args.seed).to(device)
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size,
                                                 test_batch_size=args.test_batch_size, seed=args.seed)
    optimizer = MeZO(model=model, random_mode=args.random_mode,
                     lr=args.lr, mu=args.mu, T_max=args.epochs, eta_min=args.eta_min)

    # Save this run's actual first-layer initial weights before training (for later Delta W effective-rank analysis)
    os.makedirs('./weights', exist_ok=True)
    W_init = model.fc1.weight.detach().cpu().clone()  # [hidden, in] = [128, 1024]

    best_acc = 0.0
    acc_epoch = []
    loss_epoch = []
    for epoch in range(args.epochs):
        loss = train(model, device, train_loader, optimizer, epoch, encoder)[0]
        test_acc = test(model, device, test_loader, encoder)[1]
        optimizer.update_lr()
        loss_epoch.append(loss)
        acc_epoch.append(test_acc)

    # Save the final first-layer weights after training, paired with W_init (same run, same initialization)
    W_final = model.fc1.weight.detach().cpu().clone()
    np.savez(f'./weights/weights_{args.random_mode}_SEED{args.seed}.npz',
             W_init=W_init.numpy(), W_final=W_final.numpy())

    os.makedirs('./output', exist_ok=True)
    with open(f'./output/Loss_MNIST_{args.random_mode}_SEED{args.seed}.txt', 'w') as f:
        f.write("\nAll Train Loss:\n")
        f.write(', '.join([f'{loss:.2f}' for loss in loss_epoch]))

    with open(f'./output/Accuracy_MNIST_{args.random_mode}_SEED{args.seed}.txt', 'w') as f:
        f.write("\nAll Test Accuracies:\n")
        f.write(', '.join([f'{acc:.2f}%' for acc in acc_epoch]))
