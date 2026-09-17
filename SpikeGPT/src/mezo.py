import math
import numpy as np
import torch
from src.spikingjelly.clock_driven import functional
from src.pgu import PGUXoR, PGUReuse

class MeZO:
    def __init__(self, device, model, random_mode, seed, perturb_nums, lr=0.01, epsilon=0.01, weight_decay=0,
                 lr_schedule='constant', lr_final=0.0, total_steps=None):
        self.model = model
        self.lr = lr
        self.base_lr = lr                 # initial lr (schedule anchor)
        self.lr_schedule = lr_schedule    # 'constant' | 'linear' | 'cosine'
        self.lr_final = lr_final          # end lr for linear/cosine
        self.total_steps = total_steps    # total MeZO steps over the whole run
        self.global_step = 0
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.perturb_nums = perturb_nums
        self.device = device
        self.random_mode = random_mode
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.size = sum(p.numel() for p in self.params)  # model size
        if self.random_mode == 'PGUXoR':
            self.pgu = PGUXoR(size=self.size, seed=seed, bit_width=8, device=device)
        elif self.random_mode == 'PGUReuse':
            self.pgu = PGUReuse(size=self.size, seed=seed, bit_width=8, device=device)
        else:
            print('Not using PGU random numbers')

    def _current_lr(self):
        if self.lr_schedule == 'constant' or not self.total_steps:
            return self.base_lr
        t = min(self.global_step / self.total_steps, 1.0)
        if self.lr_schedule == 'linear':
            return self.base_lr + (self.lr_final - self.base_lr) * t
        if self.lr_schedule == 'cosine':
            return self.lr_final + 0.5 * (self.base_lr - self.lr_final) * (1 + math.cos(math.pi * t))
        return self.base_lr

    def step(self, data, target):
        self.lr = self._current_lr()      # schedule sets the current lr (constant -> base_lr)
        self.global_step += 1
        with torch.no_grad():
            if self.perturb_nums > 1:
                loss_gaps = []

                if self.random_mode == 'Randn':
                    seeds = []
                    for _ in range(self.perturb_nums):
                        zo_rand_seed = np.random.randint(1000000000)
                        seeds.append(zo_rand_seed)
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(zo_rand_seed)

                        # 1. int8 (-128~127), rescale to [-0.5, 0.5]
                        # 1) normal-distributed random numbers provided by torch.normal
                        u = torch.normal(mean=0, std=1, size=(self.size,), device=self.device, generator=generator)

                        # 2. positive perturbation
                        functional.reset_net(self.model)
                        self._perturb_params(u, +self.epsilon)
                        loss_pos = self.model(data, target).item()

                        # 3. negative perturbation
                        functional.reset_net(self.model)
                        self._perturb_params(u, -2 * self.epsilon)
                        loss_neg = self.model(data, target).item()

                        # 4. restore parameters
                        self._perturb_params(u, +self.epsilon)
                        loss_gaps.append(loss_pos - loss_neg)

                    for num in range(self.perturb_nums):
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(seeds[num])
                        u = torch.normal(mean=0, std=1, size=(self.size,), device=self.device, generator=generator)
                        # 5. greadient estimation and weight update
                        grad = (loss_gaps[num]) / (2 * self.epsilon) * u
                        self._update_params(-self.lr * grad / self.perturb_nums)

                elif self.random_mode == 'Randint':
                    seeds = []
                    for _ in range(self.perturb_nums):
                        zo_rand_seed = np.random.randint(1000000000)
                        seeds.append(zo_rand_seed)
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(zo_rand_seed)

                        # 1. int8 (-128~127), rescale to [-0.5, 0.5]
                        # 2) uniform-distributed random numbers provided by torch.randint
                        u_int8 = torch.randint(-128, 128, (self.size,), dtype=torch.int8, device=self.device, generator=generator)

                        u = u_int8.float() / 256.0  # [-0.5, 0.5] (127.5/255 ≈ 0.5)
                        u = u * (12 ** 0.5)  # sqrt(12) ≈ 3.464 making variance to 1

                        # 2. positive perturbation
                        functional.reset_net(self.model)
                        self._perturb_params(u, +self.epsilon)
                        loss_pos = self.model(data, target).item()

                        # 3. negative perturbation
                        functional.reset_net(self.model)
                        self._perturb_params(u, -2 * self.epsilon)
                        loss_neg = self.model(data, target).item()

                        # 4. restore parameters
                        self._perturb_params(u, +self.epsilon)
                        loss_gaps.append(loss_pos - loss_neg)

                    for num in range(self.perturb_nums):
                        generator = torch.Generator(device=self.device)
                        generator.manual_seed(seeds[num])
                        u_int8 = torch.randint(-128, 128, (self.size,), dtype=torch.int8, device=self.device, generator=generator)
                        u = u_int8.float() / 256.0
                        u = u * (12 ** 0.5)
                        # 5. greadient estimation and weight update
                        grad = (loss_gaps[num]) / (2 * self.epsilon) * u
                        self._update_params(-self.lr * grad / self.perturb_nums)

                else:
                    self.pgu.seed_update()
                    self.pgu.reset()
                    for _ in range(self.perturb_nums):
                        # 1. int8 (-128~127), rescale to [-0.5, 0.5]
                        # 3) uniform-distributed random numbers provided by PGU
                        u_int8 = self.pgu.step()

                        u = u_int8.float() / 256.0  # [-0.5, 0.5] (127.5/255 ≈ 0.5)
                        u = u * (12 ** 0.5)  # sqrt(12) ≈ 3.464 making variance to 1

                        # 2. positive perturbation
                        functional.reset_net(self.model)
                        self._perturb_params(u, +self.epsilon)
                        loss_pos = self.model(data, target).item()

                        # 3. negative perturbation
                        functional.reset_net(self.model)
                        self._perturb_params(u, -2 * self.epsilon)
                        loss_neg = self.model(data, target).item()

                        # 4. restore parameters
                        self._perturb_params(u, +self.epsilon)
                        loss_gaps.append(loss_pos - loss_neg)

                    self.pgu.reset()
                    for num in range(self.perturb_nums):
                        u_int8 = self.pgu.step()
                        u = u_int8.float() / 256.0
                        u = u * (12 ** 0.5)
                        # 5. greadient estimation and weight update
                        grad = (loss_gaps[num]) / (2 * self.epsilon) * u
                        self._update_params(-self.lr * grad / self.perturb_nums)
            else:
                # 1. int8 (-128~127), rescale to [-0.5, 0.5]
                if self.random_mode == 'Randn':
                    # 1) normal-distributed random numbers provided by torch.normal
                    u = torch.normal(mean=0, std=1, size=(self.size,), device=self.device)
                elif self.random_mode == 'Randint':
                    # 2) uniform-distributed random numbers provided by torch.randint
                    u_int8 = torch.randint(-128, 128, (self.size,), dtype=torch.int8, device=self.device)
                    u = u_int8.float() / 128.0  # [-0.5, 0.5] (127.5/255 ≈ 0.5)
                    u = u * (3 ** 0.5)  # sqrt(12) ≈ 3.464 making variance to 1
                else:
                    # 3) uniform-distributed random numbers provided by PGU
                    u_int8 = self.pgu.step()
                    u = u_int8.float() / 128.0  # [-0.5, 0.5] (127.5/255 ≈ 0.5)
                    u = u * (3 ** 0.5)  # sqrt(12) ≈ 3.464 making variance to 1

                # 2. positive perturbation
                functional.reset_net(self.model)
                self._perturb_params(u, +self.epsilon)
                loss_pos = self.model(data, target).item()

                # 3. negative perturbation
                functional.reset_net(self.model)
                self._perturb_params(u, -2 * self.epsilon)
                loss_neg = self.model(data, target).item()

                # 4. restore parameters
                self._perturb_params(u, +self.epsilon)

                # 5. greadient estimation and weight update
                grad = (loss_pos-loss_neg) / (2 * self.epsilon) * u
                self._update_params(-self.lr * grad)

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
            p.data.mul_(1 - self.lr * self.weight_decay)
            p.data.add_(update[idx:idx + numel].view_as(p.data))
            idx += numel
