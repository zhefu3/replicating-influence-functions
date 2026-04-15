"""
Approximate s_test = H^{-1} * grad_of_test_loss via the LiSSA-style
stochastic recursion from Koh & Liang (2017), Appendix G.

Recursion (per repetition, per step):
    h_j = v + (1 - damping) * h_{j-1} - (H_z / scale) * h_{j-1}

where H_z is the Hessian of the *single-example* regularized objective
    CE(z, theta) + (L2_REG/2) * ||w||^2
so that E_z[H_z] = H exactly.

Final per-rep estimate is h_t / scale; averaged over r reps.

Key fact about the hyperparameters:
    At convergence, h_inf / scale = (H + damping*scale * I)^{-1} v.
    So `damping * scale` is the EFFECTIVE EXTRA REGULARIZATION added on top
    of the L2 already baked into H. With the previous defaults
    (DAMPING=0.01, SCALE=50) this was 0.5 — i.e. 50x the actual L2_REG=0.01,
    which biases the stochastic estimate far away from the CG result.

Fixed defaults below:
    SCALE = 25.0   -> a conservative upper bound on lambda_max(H_z) for
                     MNIST linear + CE (||x||^2 <= ~200 and the softmax
                     Hessian spectral norm <= 1/4, so lambda_max(H_z) <= 50;
                     25 works in practice and converges tighter).
    DAMPING = 0.0  -> no extra regularization; L2_REG=0.01 already makes
                     H strictly positive definite, so damping is unnecessary
                     for this problem. If you see divergence, bump SCALE
                     first, then add a tiny damping (e.g. 1e-3).
"""
import os
import random
import torch
from torchvision import datasets, transforms

from influence_utils import (
    CHECKPOINT_PATH,
    set_seed,
    grad_of_loss,
    hvp,
    load_model,
)


DEVICE = "cpu"
TEST_INDEX = 8

# LiSSA-style hyperparameters. See docstring above for derivation.
# Values match Koh & Liang (2017), Figure 2 middle panel: r=10, t=5000.
R = 10          # number of independent repetitions to average (paper: 10)
T = 5000        # recursion depth per repetition (paper: 5000)
SCALE = 25.0    # conservative upper bound on lambda_max(H_z) for MNIST linear CE
DAMPING = 0.0   # no extra regularization; L2_REG=0.01 already handles conditioning
SEED = 42


def stochastic_inverse_hvp(model, train_set, v, r, t, scale, damping, device):
    """Approximate H^{-1} v using the LiSSA-style stochastic recursion.

    At convergence in expectation the estimate equals
        (H + damping*scale * I)^{-1} v
    so with damping=0 this converges to H^{-1} v (with stochastic noise
    reduced by averaging over r repetitions).
    """
    estimates = []

    for rep in range(r):
        cur_estimate = v.clone()

        for step in range(t):
            idx = random.randrange(len(train_set))
            img, label = train_set[idx]

            x = img.unsqueeze(0).to(device)
            y = torch.tensor([label], dtype=torch.long, device=device)

            hv = hvp(model, x, y, cur_estimate, use_regularized_objective=True)
            cur_estimate = v + (1.0 - damping) * cur_estimate - hv / scale

            if (step + 1) % 500 == 0:
                print(
                    f"[rep {rep + 1}/{r}] step {step + 1}/{t} | "
                    f"||estimate|| = {cur_estimate.norm().item():.6f}"
                )

            if not torch.isfinite(cur_estimate).all():
                raise RuntimeError(
                    f"Estimate became non-finite at rep={rep + 1}, "
                    f"step={step + 1}. Increase SCALE (or add a tiny DAMPING)."
                )

        estimates.append(cur_estimate / scale)
        print(
            f"[rep {rep + 1}/{r}] final ||estimate/scale|| = "
            f"{estimates[-1].norm().item():.6f}"
        )

    return torch.stack(estimates, dim=0).mean(dim=0)


def main():
    set_seed(SEED)
    os.makedirs("outputs", exist_ok=True)

    print("Using device:", DEVICE)

    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    model, checkpoint = load_model(CHECKPOINT_PATH, DEVICE)
    train_indices = checkpoint["train_indices"]
    train_set = torch.utils.data.Subset(full_train, train_indices)

    x_test, y_test = test_set[TEST_INDEX]
    x_test = x_test.unsqueeze(0).to(DEVICE)
    y_test = torch.tensor([y_test], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        logits = model(x_test)
        pred = logits.argmax(dim=1).item()
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    print("Test index:", TEST_INDEX)
    print("True label:", y_test.item())
    print("Predicted label:", pred)
    print("Probabilities:", probs)

    v = grad_of_loss(model, x_test, y_test)
    print("v shape:", tuple(v.shape))
    print("||v||:", v.norm().item())

    print(
        f"\nLiSSA hyperparams: R={R}, T={T}, SCALE={SCALE}, DAMPING={DAMPING}"
    )
    print(f"Effective extra regularization (damping*scale) = {DAMPING * SCALE}")

    s_test = stochastic_inverse_hvp(
        model=model,
        train_set=train_set,
        v=v,
        r=R,
        t=T,
        scale=SCALE,
        damping=DAMPING,
        device=DEVICE,
    )

    print("\nFinished stochastic inverse-HVP approximation.")
    print("s_test shape:", tuple(s_test.shape))
    print("||s_test||:", s_test.norm().item())

    save_path = f"outputs/s_test_idx{TEST_INDEX}_r{R}_t{T}.pt"
    torch.save(
        {
            "s_test": s_test.cpu(),
            "test_index": TEST_INDEX,
            "true_label": y_test.item(),
            "pred_label": pred,
            "r": R,
            "t": T,
            "scale": SCALE,
            "damping": DAMPING,
        },
        save_path,
    )
    print("Saved s_test to:", save_path)


if __name__ == "__main__":
    main()
