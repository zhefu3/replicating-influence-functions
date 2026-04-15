"""
Shared utilities for the influence-function replication.

Keeps model / constants / helpers in one place so every script matches exactly.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Shared constants ----------
INPUT_DIM = 28 * 28
NUM_CLASSES = 10
L2_REG = 0.01
CHECKPOINT_PATH = "outputs/linear_mnist_lbfgs.pt"


# ---------- Shared model ----------
class LinearClassifier(nn.Module):
    """Single linear layer, no bias: parameter count = 784 * 10 = 7840."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(INPUT_DIM, NUM_CLASSES, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


# ---------- Common helpers ----------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_tensors(tensor_list):
    return torch.cat([t.reshape(-1) for t in tensor_list])


def subset_to_tensors(subset, device):
    xs, ys = [], []
    for img, label in subset:
        xs.append(img)
        ys.append(label)
    x = torch.stack(xs).to(device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y


def compute_example_loss(model, x, y):
    """Plain per-example (or per-batch mean) CE loss, no L2.

    This is what goes into the influence formula's gradient terms:
        nabla_theta L(z, theta_hat)
    The L2 regularizer only enters through the Hessian of the *training*
    objective, NOT through the per-example gradients.
    """
    logits = model(x)
    return F.cross_entropy(logits, y)


def compute_training_objective(model, x, y):
    """Regularized training objective. Its Hessian is the H used by influence.

    F.cross_entropy with a batch already returns the mean CE, so the
    Hessian of this function is:
        (1/n) * sum_i nabla^2 CE(z_i, theta) + L2_REG * I
    which exactly matches the paper's H_theta_hat.
    """
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    l2 = 0.5 * model.linear.weight.pow(2).sum()
    return ce + L2_REG * l2


def grad_of_loss(model, x, y):
    """Flat gradient of per-example CE loss w.r.t. model parameters."""
    params = [p for p in model.parameters() if p.requires_grad]
    loss = compute_example_loss(model, x, y)
    grads = torch.autograd.grad(loss, params)
    return flatten_tensors(grads).detach()


def hvp(model, x, y, vector, use_regularized_objective=True):
    """Hessian-vector product H @ vector.

    If use_regularized_objective=True (default), H is the Hessian of the
    regularized training objective on (x, y):
        H = nabla^2 [ CE_mean(x,y) + (L2_REG/2) * ||w||^2 ]
    which equals the full training Hessian when (x, y) is the full training set,
    and is an unbiased estimator of the full training Hessian when (x, y) is
    a single random sample (because L2_REG*I is a constant).
    """
    params = [p for p in model.parameters() if p.requires_grad]

    if use_regularized_objective:
        loss = compute_training_objective(model, x, y)
    else:
        loss = compute_example_loss(model, x, y)

    first_grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_first_grads = flatten_tensors(first_grads)
    grad_dot_vec = torch.dot(flat_first_grads, vector)
    hvp_tensors = torch.autograd.grad(grad_dot_vec, params)
    return flatten_tensors(hvp_tensors).detach()


def load_model(checkpoint_path=CHECKPOINT_PATH, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = LinearClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
