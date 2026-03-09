from typing import Callable, Union

import mlx.core as mx
from mlx.optimizers import AdamW, Optimizer


def zeropower_via_svd(G, steps=None) -> mx.array:
    # OPTIMIZED: Use GPU instead of CPU for SVD
    U, S, Vt = mx.linalg.svd(G)  # Let MLX choose optimal device (GPU)
    return U @ Vt


# OPTIMIZED: Removed @mx.compile to allow outer compilation to fuse better
# The full training step compilation will handle this
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7) -> mx.array:
    r"""Newton-Schulz iteration to compute the zeroth power / orthogonalization of :math:`G`. We opt to
    use a quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce :math:`UV^T` but rather something like
    :math:`US'V^T` where :math:`S'` is diagonal with :math:`S_{ii}' \sim Uniform(0.5, 1.5)`, which turns
    out not to hurt model performance at all relative to :math:`UV^T`, where :math:`USV^T = G` is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # OPTIMIZED: Skip redundant type conversion when already bfloat16
    X = G if G.dtype == mx.bfloat16 else G.astype(mx.bfloat16)

    # Normalize - using max instead of norm for potential speed (approximate)
    # norm = mx.linalg.norm(X)
    # X = X / mx.maximum(norm, eps)
    X = X / (mx.linalg.norm(X) + eps)

    if G.shape[0] > G.shape[1]:
        X = X.T

    # OPTIMIZED: Use mx.addmm for fused matrix multiply-add (like PyTorch!)
    # mx.addmm(c, a, b, alpha, beta) computes: alpha * (a @ b) + beta * c
    # This is what PyTorch uses for fast Newton-Schulz
    for _ in range(steps):
        # Compute gram matrix once
        gram = X @ X.T  # (M, M)

        # Fused operation 1: gram_update = c * (gram @ gram @ X) + b * (gram @ X)
        # This is: addmm(gram @ X, gram, gram @ X, alpha=c, beta=b)
        gram_x = gram @ X
        gram_update = mx.addmm(gram_x, gram, gram_x, alpha=c, beta=b)

        # Fused operation 2: X = a * X + gram_update
        # Could use addmm but would need identity matrix, so just do scalar multiply + add
        X = a * X + gram_update

    if G.shape[0] > G.shape[1]:
        X = X.T
    return X


class Muon(Optimizer):
    r"""MomentUm Orthogonalized by Newton-schulz [1].

    .. math::

        m_t &= \mu m_{t-1} + g_t \\
        g_t &= \mu m_t + g_t \text{ if nesterov} \\
        O_t &= \text{orthogonalize}(g_t) \\
        \theta_{t} &= \theta_{t-1} - \eta (O_t + \lambda \theta_{t-1})

    
    [1] Keller Jordan, 2024. https://github.com/KellerJordan/Muon

    Args:
        learning_rate (float or callable): The learning rate :math:`\eta`. Default: ``0.02``
        momentum (float, optional): The momentum strength :math:`\mu`. Default: ``0.95``
        nesterov (bool, optional): Enables Nesterov momentum. Default: ``True``
        backend (str, optional): The orthogonalization backend. Default: ``"newtonschulz5"``
        backend_steps (int, optional): The number of steps for orthogonalization. Default: ``5``
        alternate_optimizer (Optimizer, optional): The alternate optimizer to use when
            the parameter is not a 2D tensor. Default: ``AdamW(0.001)``

    ..
    """

    def __init__(
        self,
        learning_rate: Union[float, Callable[[mx.array], mx.array]] = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        backend: str = "newtonschulz5",
        backend_steps: int = 5,
        alternate_optimizer: Optimizer = AdamW(0.001),
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.backend = backend
        self.backend_steps = backend_steps
        self.alternate_optimizer = alternate_optimizer

        try:
            self.orthogonalize = {
                "svd": zeropower_via_svd,
                "newtonschulz5": zeropower_via_newtonschulz5,
            }[backend]
        except KeyError:
            raise ValueError(f"Unknown backend: {backend}")

    def init_single(self, parameter: mx.array, state: dict):
        if parameter.ndim != 2 or sum(parameter.shape) > 9999:
            return self.alternate_optimizer.init_single(parameter, state)
        state["muon_buf"] = mx.zeros_like(parameter)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """Apply Muon optimization update with Newton-Schulz orthogonalization."""
        lr = self.learning_rate.astype(gradient.dtype)
        if "muon_buf" not in state:
            return self.alternate_optimizer.apply_single(gradient, parameter, state)

        buf = state["muon_buf"]
        buf = buf * self.momentum + gradient
        state["muon_buf"] = buf

        gradient = (gradient + self.momentum * buf) if self.nesterov else buf
        gradient = self.orthogonalize(gradient, steps=self.backend_steps)
        gradient *= max(1, gradient.shape[0] / gradient.shape[1]) ** 0.5

        return parameter - lr * gradient
