"""
Leave-one-out verification — reproduces Figure 2 of Koh & Liang (2017).

For the top-K training points (by |predicted influence|), retrain the model
without each of them and measure the actual change in test loss at a single
chosen test point. Scatter-plot predicted vs actual.

Paper setup for Figure 2 (middle panel):
    - logistic regression on 10-class MNIST, n = 55000, L2_REG = 0.01
    - a single wrongly-classified test point z_test
    - Left panel:  top-500 by |CG influence|,        y-axis = CG influence
    - Middle panel: same top-500 as left,            y-axis = stochastic influence
    - Right panel: CNN, top-100 by |CG influence|,   y-axis = CG influence

So for the MIDDLE panel we pick top-K by the *exact* (CG) influence — that is
the same set of points as the left panel — and then on the y-axis we put the
*stochastic* approximation's predicted influence for those same points. This
is the canonical way to visually demonstrate "the stochastic approximation
tracks the exact computation." Decoupling the two knobs avoids letting
stochastic noise distort the choice of which points to retrain.

To reproduce:
    LEFT PANEL:
        SELECTION_PATH   = predicted_influence_cg_idx8.pt
        PREDICTION_PATH  = predicted_influence_cg_idx8.pt      (same file)

    MIDDLE PANEL:
        SELECTION_PATH   = predicted_influence_cg_idx8.pt      (CG picks)
        PREDICTION_PATH  = predicted_influence_idx8_r10_t5000.pt  (stoch y-axis)

Runtime note: retrains are warm-started from the baseline full-model weights,
which makes each L-BFGS fit converge in a handful of iterations instead of
re-converging from scratch.
"""
import os
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from influence_utils import (
    CHECKPOINT_PATH,
    LinearClassifier,
    set_seed,
    subset_to_tensors,
    compute_example_loss,
    compute_training_objective,
)


# ---------- What to reproduce ----------
# Middle panel of Figure 2: pick top-K by CG, put stochastic predictions on y.
SELECTION_PATH = "outputs/predicted_influence_cg_idx8.pt"
PREDICTION_PATH = "outputs/predicted_influence_idx8_r10_t5000.pt"

TOP_K = 500
SEED = 42

# Label used in plot title / output filenames so different pairings don't
# overwrite each other.
PLOT_LABEL = "middle (linear, stochastic)"
OUT_TAG = "middle"

# Force CPU: the LOO step uses float64 for numerical precision, and MPS
# does not support float64 tensor ops. CPU is also more reproducible for
# this precision-critical comparison.
TRAIN_DEVICE = "cpu"


def fit_linear_lbfgs(train_x, train_y, device, init_state_dict=None,
                     outer_steps=3, max_iter=200):
    """Fit a LinearClassifier (in float64) with L-BFGS on the given set.

    Key numerical details needed for LOO to be tight:

    1. FLOAT64 throughout. Per-point test-loss differences are ~1e-3 to 1e-5.
       Cross-entropy at loss ~3.4 in float32 has precision ~3e-7, which is
       plenty — BUT the warm-start optimum for the full data is within
       ~1e-5 of the reduced-data optimum, and the quadratic model L-BFGS
       builds lives inside that gap. In float32, line-search step sizes
       get rounded to zero and L-BFGS can terminate at the warm-start
       without ever moving. In float64 we keep ~1e-16 relative precision
       and the optimizer moves normally.

    2. tolerance_grad = 0 and tolerance_change = 0. Warm-starting from the
       full-data optimum means the initial gradient is tiny (O(1/n)), so
       with the default PyTorch tolerance 1e-7 L-BFGS triggers its stop
       condition *at iteration 0* and returns the warm-start unchanged.
       Zeroing both tolerances forces it to run until max_iter.

    3. Large max_iter + strong Wolfe line search + history_size=20. Strongly
       convex problem, so L-BFGS converges in a handful of iterations once
       it's allowed to take a first step; the extra budget is cheap.
    """
    model = LinearClassifier().to(device).double()
    if init_state_dict is not None:
        state = {k: v.to(device).double() for k, v in init_state_dict.items()}
        model.load_state_dict(state)

    train_x64 = train_x.double()
    train_y64 = train_y

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=0.0,
        tolerance_change=0.0,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    for _ in range(outer_steps):
        def closure():
            optimizer.zero_grad()
            loss = compute_training_objective(model, train_x64, train_y64)
            loss.backward()
            return loss

        optimizer.step(closure)

    return model


def evaluate_test_loss(model, x_test, y_test):
    """Cross-entropy on a single test point, computed in float64 (no L2)."""
    model.eval()
    with torch.no_grad():
        x_test64 = x_test.double() if x_test.dtype != torch.float64 else x_test
        return compute_example_loss(model, x_test64, y_test).item()


def pearson_corr(xs, ys):
    """Pearson correlation between two 1-D sequences."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    x_c = xs - xs.mean()
    y_c = ys - ys.mean()
    denom = np.sqrt((x_c ** 2).sum() * (y_c ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((x_c * y_c).sum() / denom)


def spearman_corr(xs, ys):
    """Spearman rank correlation — i.e. Pearson applied to the ranks."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    rx = xs.argsort().argsort().astype(np.float64)
    ry = ys.argsort().argsort().astype(np.float64)
    return pearson_corr(rx, ry)


def same_sign_ratio(xs, ys):
    """Fraction of index pairs where xs[i] and ys[i] share the same sign."""
    if len(xs) == 0:
        return float("nan")
    return sum(1 for x, y in zip(xs, ys)
               if (x >= 0 and y >= 0) or (x < 0 and y < 0)) / len(xs)


def main():
    set_seed(SEED)
    os.makedirs("outputs", exist_ok=True)

    print("TRAIN_DEVICE:", TRAIN_DEVICE)
    print("Selection (top-K picks) from:", SELECTION_PATH)
    print("Prediction (y-axis values) from:", PREDICTION_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    selection_bundle = torch.load(SELECTION_PATH, map_location="cpu")
    prediction_bundle = torch.load(PREDICTION_PATH, map_location="cpu")

    if selection_bundle["test_index"] != prediction_bundle["test_index"]:
        raise ValueError(
            "Selection and prediction bundles are for different test points."
        )
    test_index = selection_bundle["test_index"]
    train_indices = checkpoint["train_indices"]

    # Top-K indices come from the SELECTION bundle (exact CG).
    selection_records = selection_bundle["records_sorted"]
    selected_recs = selection_records[:TOP_K]
    selected_local_idxs = [r["train_local_index"] for r in selected_recs]

    # Look up the PREDICTION bundle's predicted_remove_diff for each of
    # those specific training points. We index by train_local_index.
    pred_by_idx = {
        r["train_local_index"]: r["predicted_remove_diff"]
        for r in prediction_bundle["records_sorted"]
    }
    missing = [i for i in selected_local_idxs if i not in pred_by_idx]
    if missing:
        raise ValueError(
            f"{len(missing)} selected indices have no entry in the "
            f"prediction bundle (first missing: {missing[0]})."
        )

    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    full_train_subset = torch.utils.data.Subset(full_train, train_indices)
    print("\nLoading full training tensors...")
    full_train_x, full_train_y = subset_to_tensors(full_train_subset, TRAIN_DEVICE)

    # Baseline model = the checkpoint weights, re-fit in float64 to exactly
    # match the precision of the LOO runs. The "actual diff" we ultimately
    # plot is (LOO loss in float64) - (baseline loss in float64), so the
    # subtraction only makes sense if both terms come from the same
    # optimizer/precision regime. Re-fitting with zero tolerances from the
    # stored checkpoint is cheap (it starts at a near-optimum) and removes
    # the risk of a mismatched baseline biasing every "actual" diff.
    baseline_state = copy.deepcopy(checkpoint["model_state_dict"])
    full_model = fit_linear_lbfgs(
        full_train_x, full_train_y, TRAIN_DEVICE,
        init_state_dict=baseline_state,
        outer_steps=3, max_iter=200,
    )
    # Keep the converged float64 full-model state as our warm-start for LOOs.
    baseline_state = {k: v.detach().clone() for k, v in full_model.state_dict().items()}

    x_test_img, y_test_int = test_set[test_index]
    x_test = x_test_img.unsqueeze(0).to(TRAIN_DEVICE)
    y_test = torch.tensor([y_test_int], dtype=torch.long, device=TRAIN_DEVICE)

    base_test_loss = evaluate_test_loss(full_model, x_test, y_test)
    print(f"Baseline test loss at index {test_index}: {base_test_loss:.10f}")
    print(f"Running leave-one-out retraining for top K={TOP_K} points")

    # Reuse a single boolean mask over training rows instead of rebuilding
    # a Subset for every iteration.
    n_train = full_train_x.size(0)
    keep_mask = torch.ones(n_train, dtype=torch.bool, device=TRAIN_DEVICE)

    results = []

    for rank, rec in enumerate(selected_recs, start=1):
        remove_local_idx = rec["train_local_index"]
        remove_orig_idx = rec["train_original_index"]
        label = rec["label"]
        predicted_remove_diff = pred_by_idx[remove_local_idx]

        keep_mask[remove_local_idx] = False
        reduced_train_x = full_train_x[keep_mask]
        reduced_train_y = full_train_y[keep_mask]

        loo_model = fit_linear_lbfgs(
            reduced_train_x,
            reduced_train_y,
            TRAIN_DEVICE,
            init_state_dict=baseline_state,
            outer_steps=3,
            max_iter=200,
        )
        loo_test_loss = evaluate_test_loss(loo_model, x_test, y_test)
        actual_remove_diff = loo_test_loss - base_test_loss

        keep_mask[remove_local_idx] = True

        results.append(
            {
                "rank": rank,
                "train_local_index": remove_local_idx,
                "train_original_index": remove_orig_idx,
                "label": label,
                "predicted_remove_diff": predicted_remove_diff,
                "actual_remove_diff": actual_remove_diff,
            }
        )

        if rank % 25 == 0 or rank == len(selected_recs):
            print(
                f"[{rank:4d}/{len(selected_recs)}] "
                f"pred={predicted_remove_diff:+.6e}  "
                f"actual={actual_remove_diff:+.6e}"
            )

        del loo_model, reduced_train_x, reduced_train_y
        if TRAIN_DEVICE == "mps":
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    save_path = f"outputs/loo_results_top{TOP_K}_{OUT_TAG}_idx{test_index}.pt"
    torch.save(
        {
            "test_index": test_index,
            "base_test_loss": base_test_loss,
            "top_k": TOP_K,
            "selection_path": SELECTION_PATH,
            "prediction_path": PREDICTION_PATH,
            "plot_label": PLOT_LABEL,
            "results": results,
        },
        save_path,
    )
    print("\nSaved results to:", save_path)

    xs = [r["actual_remove_diff"] for r in results]
    ys = [r["predicted_remove_diff"] for r in results]

    pearson = pearson_corr(xs, ys)
    spearman = spearman_corr(xs, ys)
    sign_match = same_sign_ratio(xs, ys)

    print(f"Pearson correlation:  {pearson:.4f}")
    print(f"Spearman correlation: {spearman:.4f}")
    print(f"Same-sign ratio:      {sign_match:.4f}")

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=12, alpha=0.6)
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Actual diff in test loss (leave-one-out retraining)")
    plt.ylabel("Predicted diff in test loss (influence function)")
    plt.title(
        f"Figure 2 {PLOT_LABEL} — test idx={test_index}, top-{TOP_K}\n"
        f"pearson={pearson:.3f}, sign-match={sign_match:.2f}"
    )
    plt.tight_layout()
    fig_path = f"outputs/loo_scatter_top{TOP_K}_{OUT_TAG}_idx{test_index}.png"
    plt.savefig(fig_path, dpi=150)
    print("Saved scatter plot to:", fig_path)


if __name__ == "__main__":
    main()
