"""
Quick sanity check for the Hessian-vector product implementation.

The Hessian is a linear operator, so H(a*v) must equal a*H(v) exactly (up to
floating-point error). A non-zero difference here indicates a bug in the HVP
routine — run this whenever `influence_utils.hvp` is modified.
"""
import torch
from torchvision import datasets, transforms

from influence_utils import (
    CHECKPOINT_PATH,
    load_model,
    hvp,
)


DEVICE = "cpu"


def main():
    print("Using device:", DEVICE)

    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    model, _ = load_model(CHECKPOINT_PATH, DEVICE)
    # load_model already calls eval(); keep it that way so the sanity check
    # exercises the same code path as the influence computations.

    params = [p for p in model.parameters() if p.requires_grad]
    param_dim = sum(p.numel() for p in params)
    print("Parameter dimension:", param_dim)

    # Small batch is enough to exercise the HVP code path.
    batch_size = 128
    xs = [train_set[i][0] for i in range(batch_size)]
    ys = [train_set[i][1] for i in range(batch_size)]

    x = torch.stack(xs).to(DEVICE)
    y = torch.tensor(ys, dtype=torch.long, device=DEVICE)

    torch.manual_seed(0)
    v = torch.randn(param_dim, device=DEVICE)

    hv = hvp(model, x, y, v, use_regularized_objective=True)

    print("v shape:", tuple(v.shape))
    print("Hv shape:", tuple(hv.shape))
    print("||v||:", v.norm().item())
    print("||Hv||:", hv.norm().item())

    # Linearity check: ||H(a*v) - a*H(v)|| should be ~0 up to float precision.
    a = 2.5
    hv_scaled_input = hvp(model, x, y, a * v, use_regularized_objective=True)
    diff = (hv_scaled_input - a * hv).norm().item()
    print("Linearity check ||H(av) - aH(v)||:", diff)


if __name__ == "__main__":
    main()
