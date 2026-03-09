"""
Batched Muon optimizer for MLX - processes all parameters together instead of one-by-one.

This attempts to work around MLX's apply_single() bottleneck by batching Newton-Schulz
operations across all matrix parameters.
"""

import mlx.core as mx
from mlx.optimizers import Optimizer, AdamW


def zeropower_via_newtonschulz5_batched(G_list, steps=5, eps=1e-7):
    """
    Batched Newton-Schulz: process multiple matrices at once.

    Args:
        G_list: List of gradient matrices (each is 2D)
        steps: Number of Newton-Schulz iterations
        eps: Epsilon for numerical stability

    Returns:
        List of orthogonalized gradients (same order as input)
    """
    if not G_list:
        return []

    a, b, c = (3.4445, -4.7750, 2.0315)

    results = []

    # Process each gradient (we still can't batch different shapes easily)
    # But at least they're all in one function call
    for G in G_list:
        if len(G.shape) != 2:
            results.append(G)  # Return unchanged if not 2D
            continue

        # Convert to bfloat16 if needed
        X = G if G.dtype == mx.bfloat16 else G.astype(mx.bfloat16)

        # Normalize
        X = X / (mx.linalg.norm(X) + eps)

        # Transpose if tall
        transposed = G.shape[0] > G.shape[1]
        if transposed:
            X = X.T

        # Newton-Schulz iterations
        for _ in range(steps):
            gram = X @ X.T
            X = a * X + b * (gram @ X) + c * (gram @ (gram @ X))

        # Transpose back if needed
        if transposed:
            X = X.T

        results.append(X)

    return results


class BatchedMuon(Optimizer):
    """
    Batched Muon optimizer that processes all parameters together.

    This avoids the per-parameter apply_single() bottleneck by collecting
    all matrix parameters and running Newton-Schulz on all of them in one pass.
    """

    def __init__(
        self,
        learning_rate=0.02,
        momentum=0.95,
        nesterov=True,
        backend_steps=5,
        alternate_optimizer=None,
    ):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.momentum = momentum
        self.nesterov = nesterov
        self.backend_steps = backend_steps

        # For non-2D parameters, fall back to AdamW
        if alternate_optimizer is None:
            self.alternate_optimizer = AdamW(learning_rate=0.001)
        else:
            self.alternate_optimizer = alternate_optimizer

    def init_single(self, parameter: mx.array, state: dict):
        """Initialize state for a single parameter."""
        # For 2D matrices, use Muon momentum buffer
        if parameter.ndim == 2 and sum(parameter.shape) <= 9999:
            state["muon_buf"] = mx.zeros_like(parameter)
            state["is_muon"] = True
        else:
            # Fall back to alternate optimizer
            state["is_muon"] = False
            self.alternate_optimizer.init_single(parameter, state)

    def apply_single(self, gradient: mx.array, parameter: mx.array, state: dict):
        """
        Apply update for a single parameter.

        Note: This still gets called per-parameter, but we'll override update()
        to batch everything.
        """
        lr = self.learning_rate.astype(gradient.dtype)

        if not state.get("is_muon", False):
            return self.alternate_optimizer.apply_single(gradient, parameter, state)

        # Muon update (but this won't actually be used if we override update())
        buf = state["muon_buf"]
        buf = buf * self.momentum + gradient
        state["muon_buf"] = buf

        grad_update = (gradient + self.momentum * buf) if self.nesterov else buf

        # Scale by aspect ratio
        grad_update = grad_update * (max(1, gradient.shape[0] / gradient.shape[1]) ** 0.5)

        return parameter - lr * grad_update

    def update(self, model, gradients):
        """
        Override update() to batch Newton-Schulz across all parameters.

        This is the key optimization - instead of calling apply_single() for each
        parameter, we collect all matrix parameters and process them together.
        """
        from mlx.utils import tree_flatten, tree_unflatten

        # Flatten parameters and gradients
        flat_params = tree_flatten(model.trainable_parameters())
        flat_grads = tree_flatten(gradients)

        # Separate matrix params (Muon) from non-matrix params (alternate optimizer)
        muon_indices = []
        muon_params = []
        muon_grads = []
        muon_paths = []

        for idx, ((path, param), (_, grad)) in enumerate(zip(flat_params, flat_grads)):
            if param.ndim == 2 and sum(param.shape) <= 9999:
                # This is a Muon parameter
                muon_indices.append(idx)
                muon_params.append(param)
                muon_grads.append(grad)
                muon_paths.append(path)

        # Update momentum buffers and collect gradients for batched Newton-Schulz
        grads_for_ns = []

        for path, grad, param in zip(muon_paths, muon_grads, muon_params):
            # Get or initialize state
            if path not in self.state:
                self.state[path] = {}
                self.init_single(param, self.state[path])

            state = self.state[path]

            # Update momentum buffer (ensure it matches gradient shape)
            if state["muon_buf"].shape != grad.shape:
                # Reinitialize if shape mismatch (shouldn't happen but be safe)
                state["muon_buf"] = mx.zeros_like(grad)

            buf = state["muon_buf"]
            buf = buf * self.momentum + grad
            state["muon_buf"] = buf

            # Compute gradient update (with Nesterov if enabled)
            grad_update = (grad + self.momentum * buf) if self.nesterov else buf
            grads_for_ns.append(grad_update)

        # BATCHED NEWTON-SCHULZ: Process all gradients together
        if self.backend_steps > 0:
            ortho_grads = zeropower_via_newtonschulz5_batched(grads_for_ns, steps=self.backend_steps)
        else:
            ortho_grads = grads_for_ns

        # Apply updates to Muon parameters
        lr = self.learning_rate
        for path, param, ortho_grad in zip(muon_paths, muon_params, ortho_grads):
            # Scale by aspect ratio
            scaled_grad = ortho_grad * (max(1, ortho_grad.shape[0] / ortho_grad.shape[1]) ** 0.5)

            # Ensure shapes match (reshape if needed)
            if scaled_grad.shape != param.shape:
                # This shouldn't happen, but if it does, reshape to match parameter
                # This might be due to transpose or other transformations
                if scaled_grad.size == param.size:
                    scaled_grad = scaled_grad.reshape(param.shape)
                else:
                    # Skip this update if sizes don't match at all
                    print(f"Warning: Shape mismatch for {path}: grad {scaled_grad.shape} vs param {param.shape}")
                    continue

            # Update parameter
            new_param = param - lr * scaled_grad

            # Set the new parameter value (need to navigate the tree)
            self._set_parameter(model, path, new_param)

        # Handle non-Muon parameters with alternate optimizer
        for idx, ((path, param), (_, grad)) in enumerate(zip(flat_params, flat_grads)):
            if idx not in muon_indices:
                # Use alternate optimizer for non-matrix parameters
                if path not in self.state:
                    self.state[path] = {}
                    self.alternate_optimizer.init_single(param, self.state[path])

                new_param = self.alternate_optimizer.apply_single(grad, param, self.state[path])
                self._set_parameter(model, path, new_param)

    def _set_parameter(self, model, path, value):
        """Helper to set a parameter value given its path."""
        parts = path.split('.')
        obj = model
        for part in parts[:-1]:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)

        final_key = parts[-1]
        if final_key.isdigit():
            obj[int(final_key)] = value
        else:
            setattr(obj, final_key, value)
