from abc import abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, List
from numpy import ndarray
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class Base(Optimizer):
    def __init__(self, params, idx, w, agents, lr=0.2, name=None, device=None, eps=1e-5, weight_decay=0):

        defaults = dict(idx=idx, lr=lr, w=w, agents=agents, name=name, device=device,
                        eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)

    @classmethod
    def cls_collect_params_grads(cls, optimizer: Optimizer, independent: bool = False):
        var_s = []
        grads = []
        for group in optimizer.param_groups:
            if independent:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    var_s.append(p.data.clone().detach())
                    grads.append(p.grad.data.clone().detach())
                return var_s, grads
            for p in group['params']:
                if p.grad is None:
                    continue
                var_s.append(p.data)
                grads.append(p.grad.data)
        return var_s, grads

    def collect_params_grads(self, independent: bool = False):
        return self.cls_collect_params_grads(self, independent)

    def collect_lr(self):
        for group in self.param_groups:
            return group["lr"]

    def collect_prev_lr(self):
        for group in self.param_groups:
            return group["prev_lr"]

    @property
    def _device(self) -> torch.device:
        return self.param_groups[0]["device"]

    @property
    def _w(self) -> ndarray:
        return self.param_groups[0]["w"]

    @abstractmethod
    def step(self, *args, **kwargs) -> Any:
        """step method in Optimizer class"""


class GD(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self,
             lr_constant: float,
             ) -> None:

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                p.data = p.data - lr_constant * p.grad.data
                continue

        return None


class DSGD(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def collect_grad(cls, params: Iterable) -> List[Tensor]:
        grads = []
        for p in params:
            if p.grad is None:
                continue
            grads.append(p.grad.data)
        return grads

    @classmethod
    def collect_x(cls, params: Iterable) -> List[Tensor]:
        var_x: list[Tensor] = []
        for p in params:
            if p.grad is None:
                continue
            var_x.append(p.data.clone().detach())
        return var_x

    def step(self,
             lr_list: Sequence[float],
             switching_k: int,
             lr_constant: float,
             k: int,
             vars: dict[int, list[Tensor]],
             grads: dict[int, list[Tensor]],
             ) -> None:

        for group in self.param_groups:
            idx = group['idx']
            agents = group["agents"]
            device = group["device"]
            w = group["w"]

            printed_switch_msg = False
            sub = 0
            for i, p in enumerate(group['params']):
                summat_x: Tensor = torch.zeros_like(p).to(device)
                if k == 0:
                    for j in range (agents):
                        temp_var: Tensor = vars[j][i + sub] - lr_list[j] * grads[j][i + sub]
                        summat_x += w[idx,j] * temp_var
                    p.data = summat_x
                    continue
                if switching_k is not None and k >= switching_k:
                    if not printed_switch_msg and k == switching_k:
                        print("Switch to the same stepsize", switching_k)
                        printed_switch_msg = True
                    for j in range(agents):
                        temp_var: Tensor = vars[j][i + sub] - lr_constant * grads[j][i + sub]
                        summat_x += w[idx, j] * temp_var
                    p.data = summat_x
                else:
                    for j in range(agents):
                        temp_var: Tensor = vars[j][i + sub] - lr_list[j] * grads[j][i + sub]
                        summat_x += w[idx, j] * temp_var
                    p.data = summat_x

        return None


