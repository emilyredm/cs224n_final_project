from typing import Callable, Iterable, Tuple
import math

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                print("state: ", state)
                print("group: ", group)

                ### TODO
                B1, B2 = group["betas"]
                epsilon = group["eps"]
                w_decay = group["weight_decay"]
                correct_b = group["correct_bias"]

                ## Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros(p.data.size(), dtype=p.data.dtype, device=p.data.device)
                    state["exp_avg_sq"] = torch.zeros(p.data.size(), dtype=p.data.dtype, device=p.data.device)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                ## Increase step 
                state["step"] += 1
                t = state["step"]

                ## Update biased first and second moments of the gradients  
                # 1st: 
                exp_avg = (B1 * exp_avg) + ((1 - B1) * grad)
                # 2nd: 
                exp_avg_sq = (B2 * exp_avg_sq) + ((1 - B2) * (grad * grad))

                ## Apply bias correction, efficient version 
                if correct_b: 
                    b_correction_1 = 1 - B1 ** t 
                    b_correction_2 = 1 - B2 ** t 
                    alpha_correction = alpha * (b_correction_2 ** 0.5) / b_correction_1
                else: 
                    alpha_correction = alpha 

                ## Update parameters, efficient version 
                p.data = p.data - (alpha_correction * (exp_avg / (exp_avg_sq.sqrt() + epsilon)))

                ## Apply weight decay 
                if w_decay != 0 : 
                    decay = alpha * w_decay * p.data    # use un-corrected learning rate (alpha)
                    p.data = p.data - decay
        return loss
