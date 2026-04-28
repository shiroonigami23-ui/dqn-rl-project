"""
neural_network.py
=================
A fully-connected neural network implemented from scratch using only NumPy.
Supports arbitrary layer sizes, ReLU activations, and MSE loss.

Used as the Q-network (and target network) inside the DQN agent.
"""

import numpy as np
import pickle
import os


# ──────────────────────────────────────────────────────────────────────
# Activation functions
# ──────────────────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)

def linear(x: np.ndarray) -> np.ndarray:
    return x

def linear_grad(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


# ──────────────────────────────────────────────────────────────────────
# Dense Layer
# ──────────────────────────────────────────────────────────────────────

class DenseLayer:
    """Single fully-connected layer with optional activation."""

    def __init__(self, in_features: int, out_features: int, activation: str = "relu",
                 seed: int = None):
        rng = np.random.default_rng(seed)
        # He initialisation for ReLU
        scale = np.sqrt(2.0 / in_features)
        self.W  = (rng.standard_normal((in_features, out_features)) * scale).astype(np.float32)
        self.b  = np.zeros(out_features, dtype=np.float32)

        # Velocity terms for Adam
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)
        self.t  = 0                       # step counter (Adam)

        self.activation      = activation
        self._act_fn         = relu   if activation == "relu" else linear
        self._act_grad_fn    = relu_grad if activation == "relu" else linear_grad

        # Cache for backprop
        self._x_in    = None
        self._z       = None
        self._a       = None

    # ── Forward ──────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x_in = x
        self._z    = x @ self.W + self.b
        self._a    = self._act_fn(self._z)
        return self._a

    # ── Backward ─────────────────────────────────────────────────────
    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        d_out : gradient w.r.t. output of THIS layer  (batch, out)
        Returns gradient w.r.t. input of this layer   (batch, in)
        """
        d_act = d_out * self._act_grad_fn(self._z)    # (batch, out)
        self.dW = self._x_in.T @ d_act                # (in, out)
        self.db = d_act.sum(axis=0)                   # (out,)
        d_in   = d_act @ self.W.T                     # (batch, in)
        return d_in

    # ── Adam update ──────────────────────────────────────────────────
    def adam_update(self, lr: float = 1e-3,
                    beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8) -> None:
        self.t += 1
        # W
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * self.dW**2
        mW_hat  = self.mW / (1 - beta1**self.t)
        vW_hat  = self.vW / (1 - beta2**self.t)
        self.W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        # b
        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * self.db**2
        mb_hat  = self.mb / (1 - beta1**self.t)
        vb_hat  = self.vb / (1 - beta2**self.t)
        self.b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

    # ── Params ───────────────────────────────────────────────────────
    def get_params(self):
        return {"W": self.W.copy(), "b": self.b.copy()}

    def set_params(self, params: dict) -> None:
        self.W = params["W"].copy().astype(np.float32)
        self.b = params["b"].copy().astype(np.float32)

    def copy_params_from(self, other: "DenseLayer") -> None:
        self.W = other.W.copy()
        self.b = other.b.copy()

    def num_params(self) -> int:
        return self.W.size + self.b.size

    def __repr__(self):
        in_f, out_f = self.W.shape
        return f"DenseLayer({in_f} → {out_f}, act={self.activation})"


# ──────────────────────────────────────────────────────────────────────
# Sequential Network
# ──────────────────────────────────────────────────────────────────────

class NeuralNetwork:
    """
    Sequential fully-connected network.

    Parameters
    ----------
    layer_sizes : list[int]
        Sizes of each layer including input.  e.g. [4, 128, 128, 2]
    activations : list[str] | None
        One activation per hidden/output layer. Defaults to
        relu for hidden layers, linear for output.
    seed        : int
    """

    def __init__(self, layer_sizes: list, activations: list = None, seed: int = 0):
        assert len(layer_sizes) >= 2, "Need at least input+output"
        n_layers = len(layer_sizes) - 1

        if activations is None:
            activations = ["relu"] * (n_layers - 1) + ["linear"]

        assert len(activations) == n_layers, \
            f"Expected {n_layers} activations, got {len(activations)}"

        self.layers: list[DenseLayer] = []
        for i in range(n_layers):
            self.layers.append(
                DenseLayer(
                    layer_sizes[i], layer_sizes[i + 1],
                    activation=activations[i],
                    seed=seed + i
                )
            )
        self.layer_sizes = layer_sizes
        self._loss_history: list[float] = []

    # ── Forward ──────────────────────────────────────────────────────
    def forward(self, x: np.ndarray) -> np.ndarray:
        """x shape: (batch, in_features)"""
        out = x.astype(np.float32)
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward (no gradient tracking needed for inference)."""
        return self.forward(x)

    # ── Loss ─────────────────────────────────────────────────────────
    @staticmethod
    def mse_loss(pred: np.ndarray, target: np.ndarray) -> tuple:
        """Returns (loss_scalar, d_loss/d_pred)"""
        diff  = pred - target
        loss  = float(np.mean(diff**2))
        d_out = 2 * diff / diff.size
        return loss, d_out

    # ── Backward ─────────────────────────────────────────────────────
    def backward(self, d_out: np.ndarray) -> None:
        grad = d_out
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    # ── Update ───────────────────────────────────────────────────────
    def adam_update(self, lr: float = 1e-3) -> None:
        for layer in self.layers:
            layer.adam_update(lr=lr)

    # ── Train step ───────────────────────────────────────────────────
    def train_step(self, x: np.ndarray, y: np.ndarray, lr: float = 1e-3) -> float:
        pred         = self.forward(x)
        loss, d_out  = self.mse_loss(pred, y)
        self.backward(d_out)
        self.adam_update(lr)
        self._loss_history.append(loss)
        return loss

    # ── Copy weights to another network ──────────────────────────────
    def copy_weights_to(self, other: "NeuralNetwork") -> None:
        assert len(self.layers) == len(other.layers)
        for src, dst in zip(self.layers, other.layers):
            dst.copy_params_from(src)

    # ── Soft update (Polyak averaging) ───────────────────────────────
    def soft_update_from(self, source: "NeuralNetwork", tau: float = 0.005) -> None:
        """dst = tau*src + (1-tau)*dst"""
        for src_l, dst_l in zip(source.layers, self.layers):
            dst_l.W = tau * src_l.W + (1 - tau) * dst_l.W
            dst_l.b = tau * src_l.b + (1 - tau) * dst_l.b

    # ── Serialisation ────────────────────────────────────────────────
    def save(self, path: str) -> None:
        data = {
            "layer_sizes" : self.layer_sizes,
            "activations" : [l.activation for l in self.layers],
            "params"      : [l.get_params() for l in self.layers],
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  ✓ Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "NeuralNetwork":
        with open(path, "rb") as f:
            data = pickle.load(f)
        net = cls(data["layer_sizes"], data["activations"])
        for layer, params in zip(net.layers, data["params"]):
            layer.set_params(params)
        print(f"  ✓ Model loaded ← {path}")
        return net

    # ── Info ─────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = ["=" * 50, "  NeuralNetwork Architecture", "=" * 50]
        total = 0
        for i, layer in enumerate(self.layers):
            n = layer.num_params()
            total += n
            lines.append(f"  Layer {i+1}: {layer}  [{n:,} params]")
        lines += ["-" * 50, f"  Total parameters: {total:,}", "=" * 50]
        return "\n".join(lines)

    def __repr__(self):
        return f"NeuralNetwork({self.layer_sizes})"


# ──────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    net = NeuralNetwork([4, 128, 128, 2], seed=0)
    print(net.summary())

    x  = np.random.randn(32, 4).astype(np.float32)
    y  = np.random.randn(32, 2).astype(np.float32)
    for _ in range(200):
        loss = net.train_step(x, y, lr=1e-3)
    print(f"  Final loss after 200 steps: {loss:.6f}")

    net.save("/tmp/test_net.pkl")
    net2 = NeuralNetwork.load("/tmp/test_net.pkl")
    diff = np.abs(net.predict(x) - net2.predict(x)).max()
    print(f"  Max weight diff after save/load: {diff:.2e}  ✓")
