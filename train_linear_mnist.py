"""
Train baseline logistic regression on MNIST for influence-function replication.

Matches the paper's setup for Figure 2: 55000 train / 5000 val, L2_REG=0.01,
L-BFGS with strong Wolfe line search. Saves model weights together with the
train/val index splits to outputs/linear_mnist_lbfgs.pt so that downstream
influence-function scripts operate on exactly the same training set.

After training, prints the indices of misclassified test points as a convenience
for picking a TEST_INDEX to analyze in subsequent scripts.
"""
import os
import torch
from torchvision import datasets, transforms

from influence_utils import (
    L2_REG,              # stored in the checkpoint as metadata only
    CHECKPOINT_PATH,
    LinearClassifier,
    set_seed,
    subset_to_tensors,
    compute_training_objective,
)


SEED = 42
TRAIN_SIZE = 55000       # matches paper's n for MNIST logistic regression
VAL_SIZE = 5000

# Prefer MPS for speed; fall back to CPU. (Note: downstream LOO retraining
# forces CPU because MPS does not support the float64 ops it needs.)
DEVICE = (
    "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)


@torch.no_grad()
def compute_accuracy(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def main():
    set_seed(SEED)
    os.makedirs("outputs", exist_ok=True)

    print("Using device:", DEVICE)

    transform = transforms.ToTensor()

    full_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    # Fixed split: first 55k rows go to training, next 5k to validation.
    # These exact indices are saved to the checkpoint so every downstream
    # script (CG, stochastic, LOO) operates on the identical training set.
    train_subset = torch.utils.data.Subset(full_train, list(range(TRAIN_SIZE)))
    val_subset = torch.utils.data.Subset(
        full_train, list(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE))
    )

    train_x, train_y = subset_to_tensors(train_subset, DEVICE)
    val_x, val_y = subset_to_tensors(val_subset, DEVICE)
    test_x, test_y = subset_to_tensors(test_set, DEVICE)

    print("train_x shape:", tuple(train_x.shape))
    print("val_x shape:", tuple(val_x.shape))
    print("test_x shape:", tuple(test_x.shape))

    model = LinearClassifier().to(DEVICE)

    # L-BFGS settings are standard defaults for a strongly convex problem.
    # The outer loop below runs optimizer.step() 10 times because PyTorch's
    # L-BFGS uses max_iter *inner* iterations per step(); restarting the
    # optimizer a few times is more robust than one long run.
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    print("\nStart training...\n")

    for outer_step in range(1, 11):
        def closure():
            optimizer.zero_grad()
            loss = compute_training_objective(model, train_x, train_y)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            train_loss = compute_training_objective(model, train_x, train_y).item()
            val_loss = compute_training_objective(model, val_x, val_y).item()
            test_loss = compute_training_objective(model, test_x, test_y).item()

            train_logits = model(train_x)
            val_logits = model(val_x)
            test_logits = model(test_x)

            train_acc = compute_accuracy(train_logits, train_y)
            val_acc = compute_accuracy(val_logits, val_y)
            test_acc = compute_accuracy(test_logits, test_y)

        print(
            f"step={outer_step:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"test_loss={test_loss:.6f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

    # Persist the trained weights together with the exact train/val split
    # and L2 setting so influence-function scripts can reproduce the setup.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": SEED,
            "train_indices": list(range(TRAIN_SIZE)),
            "val_indices": list(range(TRAIN_SIZE, TRAIN_SIZE + VAL_SIZE)),
            "l2_reg": L2_REG,
            "device_used": DEVICE,
        },
        CHECKPOINT_PATH,
    )
    print("\nSaved checkpoint to:", CHECKPOINT_PATH)

    # Convenience: list misclassified test indices. The influence analysis
    # in Koh & Liang (2017) uses a wrongly-classified test point, so the
    # next script (inspect_test_point.py) will pick one of these.
    with torch.no_grad():
        test_logits = model(test_x)
        test_preds = test_logits.argmax(dim=1)
        wrong_indices = (test_preds != test_y).nonzero(as_tuple=False).squeeze(1)

    wrong_indices_list = wrong_indices.detach().cpu().tolist()
    print("Number of misclassified test points:", len(wrong_indices_list))
    print("First 20 wrong test indices:", wrong_indices_list[:20])


if __name__ == "__main__":
    main()
