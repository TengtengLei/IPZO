"""
Perturbation Generation Unit (PGU)

Two implementations are provided: PGUST and PGUDT.
- PGUST is the preferred option when model parameters exceed 12M.
- Note: the first run may take extra time to download the required primitives.

Usage:
Install the package
$ python3 -m pip install galois
"""

# ============================================================
# Example: How to use PGU
# ============================================================
# 1. Import
#
#    from pgu import PGUST

# 2. Compute the total number of parameters in your model
#    (i.e., the total number of elements across all tensors).
#    This will be the "size" of the PGUST random generator.
#
#    size = sum(p.numel() for p in model.parameters())

# 3. Initialize the PGUST generator once, before training.
#    - size:  total number of random integers to generate
#    - seed: 36 bit non-zero integer, default=0x123456789
#    - bit_width: 1, 8, or 16 (controls the output precision)
#    - device: where the random numbers will live (e.g., "cuda")
#
#    random_generator = PGUST(size=size, seed=0x237548289 bit_width=8, device=device)

# 4. Each time you need fresh randomness (e.g., in MeZO optimizer),
#    simply call:
#
#    z = random_generator.step()
#
#    This will return a flat 1D tensor of int8/int16 random numbers,
#    length = size. You can then slice and reshape z to match
#    the shape of each parameter tensor.

# 5. Each time you need fresh seeds (e.g., in MeZO optimizer),
#    simply call:
#
#    random_generator.seed_update()

# 6. Each time you need reset randomness (e.g., in MeZO optimizer),
#    simply call:
#
#    random_generator.reset()


import math
import galois
import os
import pickle
import torch


class PGUST:
    def __init__(self, size, seed='0x123456789', num_rows=8, num_cols=48, num_quo_lfsr=16, bit_width=8, device=None):
        self.size = size
        self.device = device
        self.root = './poly/'
        os.makedirs(self.root, exist_ok=True)

        if bit_width == 1:
            segments_per_lfsr = 36
            stride = 1
        elif bit_width == 8:
            segments_per_lfsr = 8
            stride = 4
        elif bit_width == 16:
            segments_per_lfsr = 6
            stride = 4
        else:
            raise ValueError("bit_width must be 1,8 or 16")

        # 计算LFSR数量
        self.num_rows = num_rows
        self.num_quo_lfsr = num_quo_lfsr
        self.num_cols = num_cols
        self.num_arrays = math.ceil(size / (segments_per_lfsr * num_rows * num_cols * num_quo_lfsr))
        self.lfsr_len = 36

        self.seed_states = None
        self.quo_states = None
        self.rem_states = None

        self.seed = int(seed, 16)
        self.primitives = self._load_or_generate_primitives()
        self._init_lfsr_mask()
        self._init_lfsr_seed()

        # 初始化种子和多项式
        if not (0 < self.seed <= (1 << 36) - 1):
            raise ValueError("seed must be an integer in range 1 .. 2^36-1")

        # 分段起点索引
        self.weights = 1 << torch.arange(self.lfsr_len - 1, -1, -1, device=device)
        self.starts = torch.arange((segments_per_lfsr - 1) * stride, -1, -stride, device=device)
        self.mask_w = (1 << bit_width) - 1

        self.output_dtype = torch.int16 if bit_width == 16 else torch.int8
        self.output_buffer = torch.empty(size, dtype=self.output_dtype, device=device)

    # 加载10000个36位的本原多项式
    def _load_or_generate_primitives(self):
        cache_file = os.path.join(self.root, 'primitives_degree36_100k.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)  # 加载全部
        else:
            print("Generating 100k polynomials...")
            poly_gen = galois.primitive_polys(2, degree=self.lfsr_len)
            primitives = [next(poly_gen) for _ in range(100000)]  # 预生成足够多
            with open(cache_file, 'wb') as f:
                pickle.dump(primitives, f)
            return primitives  # 返回全部

    def _init_lfsr_mask(self):
        # 将种子转换为36位张量
        common_seed = torch.tensor(
            [(self.seed >> i) & 1 for i in range(self.lfsr_len - 1, -1, -1)],
            dtype=torch.bool,
            device=self.device
        )
        # 状态张量
        # [num_arrays, lfsr_len]
        self.seed_states = common_seed.unsqueeze(0).expand(self.num_arrays, -1).clone()
        # [num_arrays, num_quo_lfsr, lfsr_len]
        self.quo_states = common_seed.unsqueeze(0).expand(self.num_arrays, self.num_quo_lfsr, -1).clone()
        # [num_arrays, num_rows, num_cols, lfsr_len]
        self.rem_states = common_seed.unsqueeze(0).expand(
            self.num_arrays, self.num_rows, self.num_cols, -1).clone()

        # 多项式掩码
        def build_mask(poly):
            poly_int = int(poly) >> 1  # 将galois多项式转换为整数
            return torch.tensor(
                [(poly_int >> i) & 1 for i in range(self.lfsr_len - 1, -1, -1)],
                dtype=torch.bool,
                device=self.device
            )

        # 商LFSR掩码 [num_quo_lfsr, lfsr_len]
        self.quo_mask = torch.stack([
            build_mask(p) for p in self.primitives[0:self.num_quo_lfsr]
        ])

        # 每个模块的掩码扩展 [num_arrays, num_quo_lfsr, lfsr_len]
        self.quo_masks = self.quo_mask.unsqueeze(0).expand(self.num_arrays, self.num_quo_lfsr, -1)

        # 余数LFSR掩码 [num_rows, num_cols, lfsr_len]
        rem_polys = self.primitives[self.num_quo_lfsr:self.num_quo_lfsr + self.num_rows * self.num_cols]
        self.rem_mask = torch.zeros(
            (self.num_rows, self.num_cols, self.lfsr_len),
            dtype=torch.bool,
            device=self.device
        )
        # 按列优先顺序填充多项式掩码
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                idx = col * self.num_rows + row
                self.rem_mask[row, col] = build_mask(rem_polys[idx])

        # 每个模块的掩码扩展 [num_arrays, num_rows, num_cols, lfsr_len]
        self.rem_masks = self.rem_mask.unsqueeze(0).expand(
            self.num_arrays, self.num_rows, self.num_cols, -1)

        # seed LFSR掩码 [num_arrays, lfsr_len]
        seed_polys = self.primitives[self.num_quo_lfsr + self.num_rows * self.num_cols:]
        self.seed_masks = torch.stack([
            build_mask(p) for p in seed_polys[:self.num_arrays]
        ])

    @staticmethod
    def _galois_lfsr_function(states, masks):
        feedback = states[..., -1:]
        shifted = torch.roll(states, shifts=1, dims=-1)
        shifted[..., 0] = False
        return shifted ^ (feedback * masks)

    def _lfsr_update(self):
        self.quo_states = self._galois_lfsr_function(self.quo_states, self.quo_masks)
        self.rem_states = self._galois_lfsr_function(self.rem_states, self.rem_masks)

    def _init_lfsr_seed(self):
        for _ in range(500):
            self.seed_update()
        self.reset()

    def _xor_segment(self):
        xor_result = self.quo_states[:, :, None, None, :] ^ self.rem_states[:, None, :, :, :]
        xor_result_int = (xor_result.to(torch.int8) * self.weights).sum(dim=-1)
        segments = xor_result_int.unsqueeze(-1) >> self.starts & self.mask_w
        return segments.reshape(-1)[:self.size].to(self.output_dtype)

    def seed_update(self):
        self.seed_states = self._galois_lfsr_function(self.seed_states, self.seed_masks)

    def reset(self):
        # seed_states [num_arrays, lfsr_len]
        self.quo_states[:] = self.seed_states.unsqueeze(1).expand(self.num_arrays, self.num_quo_lfsr, -1)
        self.rem_states[:] = self.seed_states.unsqueeze(1).unsqueeze(2).expand(
            self.num_arrays, self.num_rows, self.num_cols, -1)
        for _ in range(200):
            self._lfsr_update()

    def step(self):
        self._lfsr_update()
        random = self._xor_segment()
        output = self.output_buffer[:random.shape[0]].copy_(random)
        return output


class PGUDT:
    def __init__(self, size, seed='0x123456789', num_rows=8, num_cols=48, num_quo_lfsr=16, bit_width=8, device=None):
        self.size = size
        self.device = device
        self.root = './poly/'
        os.makedirs(self.root, exist_ok=True)

        if bit_width == 1:
            segments_per_lfsr = 36
            stride = 1
        elif bit_width == 8:
            segments_per_lfsr = 8
            stride = 4
        elif bit_width == 16:
            segments_per_lfsr = 6
            stride = 4
        else:
            raise ValueError("bit_width must be 1,8 or 16")

        # 计算LFSR数量
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_quo_lfsr = num_quo_lfsr
        self.num_arrays = math.ceil(size / (segments_per_lfsr * num_rows * num_cols * num_quo_lfsr))
        self.lfsr_len = 36

        self.seed_state = None
        self.quo_states = None
        self.rem_states = None

        self.seed = int(seed, 16)
        self.primitives = self._load_or_generate_primitives()
        self._init_lfsr_mask()

        # 初始化种子和多项式
        if not (0 < self.seed <= (1 << 36) - 1):
            raise ValueError("seed must be an integer in range 1 .. 2^36-1")

        # 分段起点索引
        self.weights = 1 << torch.arange(self.lfsr_len - 1, -1, -1, device=device)
        self.starts = torch.arange((segments_per_lfsr - 1) * stride, -1, -stride, device=device)
        self.mask_w = (1 << bit_width) - 1

        self.output_dtype = torch.int16 if bit_width == 16 else torch.int8
        self.output_buffer = torch.empty(size, dtype=self.output_dtype, device=device)

    # 加载10000个36位的本原多项式
    def _load_or_generate_primitives(self):
        cache_file = os.path.join(self.root, 'primitives_degree36_100k.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)  # 加载全部
        else:
            print("Generating 100k polynomials...")
            poly_gen = galois.primitive_polys(2, degree=self.lfsr_len)
            primitives = [next(poly_gen) for _ in range(100000)]  # 预生成足够多
            with open(cache_file, 'wb') as f:
                pickle.dump(primitives, f)
            return primitives  # 返回全部

    def _init_lfsr_mask(self):
        # 将种子转换为36位张量
        common_seed = torch.tensor(
            [(self.seed >> i) & 1 for i in range(self.lfsr_len - 1, -1, -1)],
            dtype=torch.bool,
            device=self.device
        )
        # 状态张量
        self.seed_state = common_seed.clone()
        self.quo_states = common_seed.unsqueeze(0).expand(self.num_quo_lfsr, -1).clone()
        self.rem_states = common_seed.unsqueeze(0).expand(
            self.num_arrays, self.num_rows, self.num_cols, -1).clone()

        # 多项式掩码
        def build_mask(poly):
            poly_int = int(poly) >> 1  # 将galois多项式转换为整数
            return torch.tensor(
                [(poly_int >> i) & 1 for i in range(self.lfsr_len - 1, -1, -1)],
                dtype=torch.bool,
                device=self.device
            )

        # 商LFSR掩码 [num_quo_lfsr,lfsr_len]
        self.quo_masks = torch.stack([
            build_mask(p) for p in self.primitives[0:self.num_quo_lfsr]
        ])

        # 余数LFSR掩码 [num_arrays, num_rows,num_cols,lfsr_len]
        rem_polys = self.primitives[
                    self.num_quo_lfsr:self.num_quo_lfsr + self.num_arrays * self.num_rows * self.num_cols]
        self.rem_masks = torch.zeros(
            (self.num_arrays, self.num_rows, self.num_cols, self.lfsr_len),
            dtype=torch.bool,
            device=self.device
        )
        # 按列优先顺序填充多项式掩码
        for array in range(self.num_arrays):
            for row in range(self.num_rows):
                for col in range(self.num_cols):
                    idx = col * self.num_rows + row + array * self.num_rows * self.num_cols
                    self.rem_masks[array, row, col] = build_mask(rem_polys[idx])

        # seed LFSR掩码 [1, lfsr_len]
        seed_poly = self.primitives[self.num_quo_lfsr + self.num_arrays * self.num_rows * self.num_cols]
        self.seed_mask = torch.stack([
            build_mask(seed_poly)
        ])

    @staticmethod
    def _galois_lfsr_function(states, masks):
        feedback = states[..., -1:]
        shifted = torch.roll(states, shifts=1, dims=-1)
        shifted[..., 0] = False
        return shifted ^ (feedback * masks)

    def _lfsr_update(self):
        self.quo_states = self._galois_lfsr_function(self.quo_states, self.quo_masks)
        self.rem_states = self._galois_lfsr_function(self.rem_states, self.rem_masks)

    def _xor_segment(self):
        xor_result = self.quo_states[None, :, None, None, :] ^ self.rem_states[:, None, :, :, :]
        xor_result_int = (xor_result.to(torch.int8) * self.weights).sum(dim=-1)
        segments = xor_result_int.unsqueeze(-1) >> self.starts & self.mask_w
        return segments.reshape(-1)[:self.size].to(self.output_dtype)

    def seed_update(self):
        self.seed_state = self._galois_lfsr_function(self.seed_state, self.seed_mask)

    def reset(self):
        self.quo_states[:] = self.seed_state.unsqueeze(0).unsqueeze(1).expand(self.num_arrays, self.num_quo_lfsr, -1)
        self.rem_states[:] = self.seed_state.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(
            self.num_arrays, self.num_rows, self.num_cols, -1)
        for _ in range(100):
            self._lfsr_update()

    def step(self):
        self._lfsr_update()
        random = self._xor_segment()
        output = self.output_buffer[:random.shape[0]].copy_(random)
        return output



class PGUPeZO:
    def __init__(self, size, seed='0x123456789', num_rows=8, num_cols=48, num_quo_lfsr=16, bit_width=8, device=None):
        self.size = size
        self.device = device
        self.root = './poly/'
        os.makedirs(self.root, exist_ok=True)

        if bit_width == 1:
            segments_per_lfsr = 36
            stride = 1
        elif bit_width == 8:
            segments_per_lfsr = 8  # 1*28+8=36, stride=1; 2*14+8=36, stride=2; 4*7+8=36, stride=4; 7*4+8=36, stride=7
            stride = 4
        elif bit_width == 16:
            segments_per_lfsr = 6  # 4*5+16=36, stride=4
            stride = 4
        else:
            raise ValueError("bit_width must be 1,8 or 16")

        # 计算LFSR数量
        self.num_rows = num_rows
        self.num_quo_lfsr = num_quo_lfsr
        self.num_cols = num_cols
        self.num_arrays = math.ceil(size / (segments_per_lfsr * num_rows * num_quo_lfsr * num_cols))
        self.lfsr_len = 36

        # 初始化种子和多项式
        self.seed = int(seed, 16)
        self.primitives = self._load_or_generate_primitives()
        self._init_lfsr()
        self._reseed()

        seq_row = torch.arange(num_rows)
        shift_row = torch.arange(num_quo_lfsr)
        index = (seq_row.unsqueeze(0) - shift_row.unsqueeze(1)) % num_rows
        self.index = index.flatten()
        self.shift_amount = 0

        # 分段起点索引
        self.starts = torch.arange((segments_per_lfsr - 1) * stride, -1, -stride, device=device)
        self.mask_w = (1 << bit_width) - 1

        self.output_dtype = torch.int16 if bit_width == 16 else torch.int8
        self.output_buffer = torch.empty(size, dtype=self.output_dtype, device=device)

    # 加载10000个36位的本原多项式
    def _load_or_generate_primitives(self):
        cache_file = os.path.join(self.root, 'primitives_degree36_100k.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)  # 加载全部
        else:
            print("Generating 100k polynomials...")
            poly_gen = galois.primitive_polys(2, degree=self.lfsr_len)
            primitives = [next(poly_gen) for _ in range(100000)]  # 预生成足够多
            with open(cache_file, 'wb') as f:
                pickle.dump(primitives, f)
            return primitives  # 返回全部

    def _init_lfsr(self):
        # 状态张量初始化（存储为整数）
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

        # 多项式掩码
        def build_mask(poly):
            poly_int = int(poly) >> 1  # 去掉最高位 将galois多项式转换为整数
            return poly_int

        # 余数LFSR掩码 [num_rows, num_cols]
        rem_polys = self.primitives[0:self.num_rows * self.num_cols]
        self.rem_mask = torch.zeros(
            (self.num_rows, self.num_cols),
            dtype=torch.int64,
            device=self.device
        )
        # 按列优先顺序填充多项式掩码
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                idx = col * self.num_rows + row
                self.rem_mask[row, col] = build_mask(rem_polys[idx])

        # 每个模块的掩码扩展 [num_arrays, num_rows, num_cols]
        self.rem_masks = self.rem_mask.unsqueeze(0).expand(
            self.num_arrays, self.num_rows, self.num_cols)

        # seed LFSR掩码 [num_arrays]
        seed_polys = self.primitives[self.num_rows * self.num_cols:]
        self.seed_masks = torch.tensor(
            [build_mask(p) for p in seed_polys[:self.num_arrays]],
            dtype=torch.int64,
            device=self.device
        )

    @staticmethod
    def _galois_lfsr_step(states, masks):
        feedback = states & 1  # 获取最低位
        shifted = states >> 1  # 虽然是int64类型，算术右移，但第37位为0，逻辑右移后第36位为0，然后根据反馈和掩码得到更新的最高位
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
        return segments.reshape(-1)[:self.size].to(self.output_dtype)

    def step(self):
        self._update()
        self.shift_amount = -((self.shift_amount + 1) % (self.num_quo_lfsr * self.num_rows))
        random = self._shift_segment()
        output = self.output_buffer[:random.shape[0]].copy_(random)
        return output
