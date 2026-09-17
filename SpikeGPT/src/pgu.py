"""
Perturbation Generation Unit (PGU)
==================================

Hardware-inspired pseudo-random number generators built from banks of
Galois LFSRs (linear-feedback shift registers). They produce the
perturbation vectors used by zeroth-order (MeZO) optimization.

Two implementations are provided:

  * PGUXoR   -- combines a bank of "quotient" LFSRs with a bank of
                "remainder" LFSRs via XOR. Higher randomness quality;
                preferred when the model has more than ~12M parameters.

  * PGUReuse -- keeps only the remainder bank and reuses its states via
                cyclic shifts (cheaper, fewer LFSRs).

Both emit a flat int8/int16 stream of a requested length ("size"), whose
values are drawn from uniformly-distributed LFSR outputs.

Requirements
------------
    python3 -m pip install galois

Note: the first run generates 100k degree-36 primitive polynomials and
caches them under ./poly/, which can take a while.
"""

# ============================================================
# Example: using a PGU in a MeZO-style loop
# ============================================================
# 1. Import one of the generators:
#
#        from pgu import PGUXoR
#
# 2. Compute the total number of parameters to perturb -- this is the
#    "size", i.e. the total number of random integers to generate:
#
#        size = sum(p.numel() for p in model.parameters())
#
# 3. Create the generator once, before training:
#        - size:      number of random integers produced per draw
#        - seed:      non-zero 36-bit integer (default 0x123456789)
#        - bit_width: 1, 8, or 16 (output precision)
#        - device:    where the tensors live (e.g. "cuda")
#
#        rng = PGUXoR(size=size, seed=0x237548289, bit_width=8, device=device)
#
# 4. Draw a fresh perturbation. Returns a flat int8/int16 tensor of
#    length `size`; slice/reshape it to match each parameter tensor:
#
#        z = rng.step()
#
# 5. Advance the seed LFSRs (start a new independent stream):
#
#        rng.seed_update()
#
# 6. Reset the state back to the current seed:
#
#        rng.reset()


import math
import galois
import os
import pickle
import torch


class PGUXoR:
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

        # number of parallel LFSR arrays
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

        # validate the seed range
        if not (0 < self.seed <= (1 << 36) - 1):
            raise ValueError("seed must be an integer in range 1 .. 2^36-1")

        # bit weights (MSB-first) and per-segment start bit offsets
        self.weights = 1 << torch.arange(self.lfsr_len - 1, -1, -1, device=device)
        self.starts = torch.arange((segments_per_lfsr - 1) * stride, -1, -stride, device=device)
        self.mask_w = (1 << bit_width) - 1

        self.output_dtype = torch.int16 if bit_width == 16 else torch.int8
        self.output_buffer = torch.empty(size, dtype=self.output_dtype, device=device)

    # load 100k precomputed degree-36 primitive polynomials
    def _load_or_generate_primitives(self):
        cache_file = os.path.join(self.root, 'primitives_degree36_100k.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)  # load all
        else:
            print("Generating 100k polynomials...")
            poly_gen = galois.primitive_polys(2, degree=self.lfsr_len)
            primitives = [next(poly_gen) for _ in range(100000)]  # pre-generate enough
            with open(cache_file, 'wb') as f:
                pickle.dump(primitives, f)
            return primitives  # return all

    def _init_lfsr_mask(self):
        # expand the seed into a 36-bit tensor
        common_seed = torch.tensor(
            [(self.seed >> i) & 1 for i in range(self.lfsr_len - 1, -1, -1)],
            dtype=torch.bool,
            device=self.device
        )
        # state tensors
        # [num_arrays, lfsr_len]
        self.seed_states = common_seed.unsqueeze(0).expand(self.num_arrays, -1).clone()
        # [num_arrays, num_quo_lfsr, lfsr_len]
        self.quo_states = common_seed.unsqueeze(0).expand(self.num_arrays, self.num_quo_lfsr, -1).clone()
        # [num_arrays, num_rows, num_cols, lfsr_len]
        self.rem_states = common_seed.unsqueeze(0).expand(
            self.num_arrays, self.num_rows, self.num_cols, -1).clone()

        # polynomial (feedback) masks
        def build_mask(poly):
            poly_int = int(poly) >> 1  # convert the galois polynomial to an integer (drop the leading term)
            return torch.tensor(
                [(poly_int >> i) & 1 for i in range(self.lfsr_len - 1, -1, -1)],
                dtype=torch.bool,
                device=self.device
            )

        # quotient-LFSR masks [num_quo_lfsr, lfsr_len]
        self.quo_mask = torch.stack([
            build_mask(p) for p in self.primitives[0:self.num_quo_lfsr]
        ])

        # broadcast masks across all arrays [num_arrays, num_quo_lfsr, lfsr_len]
        self.quo_masks = self.quo_mask.unsqueeze(0).expand(self.num_arrays, self.num_quo_lfsr, -1)

        # remainder-LFSR masks [num_rows, num_cols, lfsr_len]
        rem_polys = self.primitives[self.num_quo_lfsr:self.num_quo_lfsr + self.num_rows * self.num_cols]
        self.rem_mask = torch.zeros(
            (self.num_rows, self.num_cols, self.lfsr_len),
            dtype=torch.bool,
            device=self.device
        )
        # fill the masks in column-major order
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                idx = col * self.num_rows + row
                self.rem_mask[row, col] = build_mask(rem_polys[idx])

        # broadcast masks across all arrays [num_arrays, num_rows, num_cols, lfsr_len]
        self.rem_masks = self.rem_mask.unsqueeze(0).expand(
            self.num_arrays, self.num_rows, self.num_cols, -1)

        # seed-LFSR masks [num_arrays, lfsr_len]
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

    def seed_update(self):
        self.seed_states = self._galois_lfsr_function(self.seed_states, self.seed_masks)

    def _init_lfsr_seed(self):
        for _ in range(500):
            self.seed_update()
        self.reset()

    def reset(self):
        # seed_states [num_arrays, lfsr_len]
        self.quo_states[:] = self.seed_states.unsqueeze(1).expand(self.num_arrays, self.num_quo_lfsr, -1)
        self.rem_states[:] = self.seed_states.unsqueeze(1).unsqueeze(2).expand(
            self.num_arrays, self.num_rows, self.num_cols, -1)
        for _ in range(200):
            self._lfsr_update()

    def _xor_segment(self):
        xor_result = self.quo_states[:, :, None, None, :] ^ self.rem_states[:, None, :, :, :]
        xor_result_int = (xor_result.to(torch.int8) * self.weights).sum(dim=-1)
        segments = xor_result_int.unsqueeze(-1) >> self.starts & self.mask_w
        return segments.reshape(-1)[:self.size].to(self.output_dtype)

    def step(self):
        self._lfsr_update()
        random = self._xor_segment()
        output = self.output_buffer[:random.shape[0]].copy_(random)
        return output


class PGUReuse:
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

        # number of parallel LFSR arrays
        self.num_rows = num_rows
        self.num_quo_lfsr = num_quo_lfsr
        self.num_cols = num_cols
        self.num_arrays = math.ceil(size / (segments_per_lfsr * num_rows * num_quo_lfsr * num_cols))
        self.lfsr_len = 36

        # seed and polynomials
        self.seed = int(seed, 16)
        self.primitives = self._load_or_generate_primitives()
        self._init_lfsr_mask()
        self._init_lfsr_seed()

        seq_row = torch.arange(num_rows)
        shift_row = torch.arange(num_quo_lfsr)
        index = (seq_row.unsqueeze(0) - shift_row.unsqueeze(1)) % num_rows
        self.index = index.flatten()
        self.shift_amount = 0

        # per-segment start bit offsets
        self.starts = torch.arange((segments_per_lfsr - 1) * stride, -1, -stride, device=device)
        self.mask_w = (1 << bit_width) - 1

        self.output_dtype = torch.int16 if bit_width == 16 else torch.int8
        self.output_buffer = torch.empty(size, dtype=self.output_dtype, device=device)

    # load 100k precomputed degree-36 primitive polynomials
    def _load_or_generate_primitives(self):
        cache_file = os.path.join(self.root, 'primitives_degree36_100k.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)  # load all
        else:
            print("Generating 100k polynomials...")
            poly_gen = galois.primitive_polys(2, degree=self.lfsr_len)
            primitives = [next(poly_gen) for _ in range(100000)]  # pre-generate enough
            with open(cache_file, 'wb') as f:
                pickle.dump(primitives, f)
            return primitives  # return all

    def _init_lfsr_mask(self):
        # state tensors (stored as packed integers)
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

        # polynomial (feedback) masks
        def build_mask(poly):
            poly_int = int(poly) >> 1  # drop the leading term; convert the galois polynomial to an integer
            return poly_int

        # remainder-LFSR masks [num_rows, num_cols]
        rem_polys = self.primitives[0:self.num_rows * self.num_cols]
        self.rem_mask = torch.zeros(
            (self.num_rows, self.num_cols),
            dtype=torch.int64,
            device=self.device
        )
        # fill the masks in column-major order
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                idx = col * self.num_rows + row
                self.rem_mask[row, col] = build_mask(rem_polys[idx])

        # broadcast masks across all arrays [num_arrays, num_rows, num_cols]
        self.rem_masks = self.rem_mask.unsqueeze(0).expand(
            self.num_arrays, self.num_rows, self.num_cols)

        # seed-LFSR masks [num_arrays]
        seed_polys = self.primitives[self.num_rows * self.num_cols:]
        self.seed_masks = torch.tensor(
            [build_mask(p) for p in seed_polys[:self.num_arrays]],
            dtype=torch.int64,
            device=self.device
        )

    @staticmethod
    def _galois_lfsr_step(states, masks):
        feedback = states & 1  # take the lowest bit
        shifted = states >> 1  # int64 arithmetic shift; bit 37 is 0 so this acts as a logical shift (bit 36 -> 0), then feedback & mask set the new top bit
        return shifted ^ (feedback * masks)

    def _lfsr_update(self):
        self.rem_states = self._galois_lfsr_step(self.rem_states, self.rem_masks)

    def seed_update(self):
        self.seed_states = self._galois_lfsr_step(self.seed_states, self.seed_masks)

    def _init_lfsr_seed(self):
        for _ in range(400):
            self.seed_update()
        self.reset()

    def reset(self):
        # seed_states [num_arrays]
        self.rem_states[:] = self.seed_states.unsqueeze(1).unsqueeze(2).expand(
            self.num_arrays, self.num_rows, self.num_cols)
        self.shift_amount = 0

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
        self._lfsr_update()
        self.shift_amount = -((self.shift_amount + 1) % (self.num_quo_lfsr * self.num_rows))
        random = self._shift_segment()
        output = self.output_buffer[:random.shape[0]].copy_(random)
        return output
