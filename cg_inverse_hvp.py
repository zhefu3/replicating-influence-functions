"""
Compute s_test = H^{-1} * grad_of_test_loss using Conjugate Gradient.

H is the Hessian of the regularized training objective, so this gives the
*exact* (up to CG tolerance) inverse-HVP used in the paper's influence formula.
"""
import os
import torch
from torchvision import datasets, transforms

from influence_utils import (
    CHECKPOINT_PATH,
    set_seed,
    subset_to_tensors,
    grad_of_loss,
    hvp,
    load_model,
)


DEVICE = "cpu"
TEST_INDEX = 8
SEED = 42

# Parameter dim is 784 * 10 = 7840. CG in exact arithmetic converges in at most
# dim steps; 30 was almost certainly too few. 200 gives CG room to actually
# reach the 1e-8 residual tolerance on a well-conditioned (L2-regularized)
# Hessian. Early stopping via TOL still kicks in once converged.
MAX_CG_ITERS = 200
TOL = 1e-8


def conjugate_gradient(hvp_fn, b, max_iters=200, tol=1e-8):
    """Solve H x = b using CG, where hvp_fn(v) returns H @ v."""
    x = torch.zeros_like(b)
    r = b.clone()  # x starts at 0, so r = b - A x = b
    p = r.clone()
    rs_old = torch.dot(r, r)

    print(f"Initial residual norm: {torch.sqrt(rs_old).item():.6e}")

    for k in range(max_iters):
        Ap = hvp_fn(p)
        denom = torch.dot(p, Ap)

        if abs(denom.item()) < 1e-20:
            print(f"CG stopped early at iter {k + 1}: denominator too small")
            break

        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap

        rs_new = torch.dot(r, r)
        residual_norm = torch.sqrt(rs_new).item()

        if (k + 1) % 10 == 0 or residual_norm < tol:
            print(f"CG iter {k + 1:03d} | residual norm = {residual_norm:.6e}")

        if residual_norm < tol:
            print(f"CG converged at iter {k + 1}")
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    else:
        # Python for-else: this block only runs if the `for` loop completed
        # without hitting `break` — i.e. CG ran out of iterations before
        # the residual dropped below `tol`.
        print(
            f"WARNING: CG did not reach tol={tol:.1e} in {max_iters} iters; "
            f"final residual norm = {residual_norm:.6e}"
        )

    return x


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
    train_subset = torch.utils.data.Subset(full_train, train_indices)

    print("Loading full training tensors...")
    train_x, train_y = subset_to_tensors(train_subset, DEVICE)

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

    def hvp_fn(vec):
        return hvp(model, train_x, train_y, vec, use_regularized_objective=True)

    s_test_cg = conjugate_gradient(
        hvp_fn=hvp_fn,
        b=v,
        max_iters=MAX_CG_ITERS,
        tol=TOL,
    )

    print("\nFinished CG inverse-HVP approximation.")
    print("s_test_cg shape:", tuple(s_test_cg.shape))
    print("||s_test_cg||:", s_test_cg.norm().item())

    save_path = f"outputs/s_test_cg_idx{TEST_INDEX}.pt"
    torch.save(
        {
            "s_test": s_test_cg.cpu(),
            "test_index": TEST_INDEX,
            "true_label": y_test.item(),
            "pred_label": pred,
            "max_cg_iters": MAX_CG_ITERS,
            "tol": TOL,
        },
        save_path,
    )
    print("Saved CG s_test to:", save_path)


if __name__ == "__main__":
    main()
