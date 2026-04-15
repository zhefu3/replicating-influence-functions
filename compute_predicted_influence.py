"""
Given a precomputed s_test = H^{-1} * grad_of_test_loss, compute the
predicted influence of every training point on the chosen test point.

Paper formulas:
    I_up,loss(z, z_test) = - grad_test^T H^{-1} grad_z = - s_test . grad_z
    Predicted change in test loss if z is removed (eps = -1/n):
        Delta L_test ~ -(1/n) * I_up,loss(z, z_test)

Switch the method used to compute s_test by changing S_TEST_PATH below.
The script auto-names its output after the source s_test file so the CG
and stochastic pipelines do not overwrite each other.
"""
import os
import torch
from torchvision import datasets, transforms

from influence_utils import (
    CHECKPOINT_PATH,
    grad_of_loss,
    load_model,
)


# Point this to whichever s_test you want to turn into predictions.
#   outputs/s_test_cg_idx8.pt            -> left panel of Figure 2
#   outputs/s_test_idx8_r10_t5000.pt     -> middle panel of Figure 2
S_TEST_PATH = "outputs/s_test_cg_idx8.pt"
DEVICE = "cpu"


def main():
    os.makedirs("outputs", exist_ok=True)
    print("Using device:", DEVICE)
    print("Loading s_test from:", S_TEST_PATH)

    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    model, checkpoint = load_model(CHECKPOINT_PATH, DEVICE)
    s_test_bundle = torch.load(S_TEST_PATH, map_location=DEVICE)

    train_indices = checkpoint["train_indices"]
    train_set = torch.utils.data.Subset(full_train, train_indices)

    n_train = len(train_set)
    print("Number of training points used by the model:", n_train)

    s_test = s_test_bundle["s_test"].to(DEVICE)
    test_index = s_test_bundle["test_index"]

    print("Loaded s_test for test index:", test_index)
    print("s_test shape:", tuple(s_test.shape))
    print("||s_test||:", s_test.norm().item())

    records = []

    for i in range(n_train):
        img, label = train_set[i]
        x = img.unsqueeze(0).to(DEVICE)
        y = torch.tensor([label], dtype=torch.long, device=DEVICE)

        train_grad = grad_of_loss(model, x, y)

        # I_up,loss = - s_test . grad_z
        influence_upweight = -torch.dot(s_test, train_grad).item()
        # Predicted change in test loss when removing point z (eps = -1/n):
        #   Delta L_test ~ -(1/n) * I_up,loss
        predicted_remove_diff = -influence_upweight / n_train

        records.append(
            {
                "train_local_index": i,
                "train_original_index": train_indices[i],
                "label": int(label),
                "influence_upweight": influence_upweight,
                "predicted_remove_diff": predicted_remove_diff,
            }
        )

        if (i + 1) % 5000 == 0:
            print(f"Processed {i + 1}/{n_train} training points")

    records_sorted = sorted(
        records,
        key=lambda d: abs(d["influence_upweight"]),
        reverse=True,
    )

    # Display cap for the sanity print only; does NOT affect saved data.
    # The full sorted list of all n_train records is saved below.
    PRINT_TOP_K = 20
    print(f"\nTop {PRINT_TOP_K} training points by |influence_upweight|:")
    for rank, rec in enumerate(records_sorted[:PRINT_TOP_K], start=1):
        print(
            f"rank={rank:02d} | "
            f"local_idx={rec['train_local_index']} | "
            f"orig_idx={rec['train_original_index']} | "
            f"label={rec['label']} | "
            f"influence_upweight={rec['influence_upweight']:.6f} | "
            f"predicted_remove_diff={rec['predicted_remove_diff']:.8f}"
        )

    # Derive the output filename from the input s_test filename so CG
    # and stochastic runs don't overwrite each other.
    s_test_basename = os.path.splitext(os.path.basename(S_TEST_PATH))[0]
    tag = s_test_basename.replace("s_test_", "")
    save_path = f"outputs/predicted_influence_{tag}.pt"
    torch.save(
        {
            "test_index": test_index,
            "n_train": n_train,
            "records_sorted": records_sorted,
            "source_s_test_path": S_TEST_PATH,
        },
        save_path,
    )
    print("\nSaved predicted influence records to:", save_path)


if __name__ == "__main__":
    main()
