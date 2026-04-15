"""
Visualize one test point and confirm the model's prediction / loss / gradient.

Used to pick a TEST_INDEX for the influence analysis. Koh & Liang (2017) do
their demonstration on a *misclassified* test point, because that is where the
question "which training points pushed this prediction wrong?" is meaningful.

TEST_INDEX = 8 is our chosen point: true label 5, model predicts 6. The
checkpoint's training run prints the list of wrongly-classified test indices
at the end, which is where this default came from.
"""
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from influence_utils import (
    CHECKPOINT_PATH,
    load_model,
    grad_of_loss,
)


TEST_INDEX = 8
DEVICE = "cpu"


def main():
    print("Using device:", DEVICE)

    transform = transforms.ToTensor()
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    model, _ = load_model(CHECKPOINT_PATH, DEVICE)

    x, y = test_set[TEST_INDEX]
    x = x.unsqueeze(0).to(DEVICE)  # (1, 1, 28, 28)
    y = torch.tensor([y], dtype=torch.long, device=DEVICE)

    logits = model(x)
    pred = logits.argmax(dim=1).item()
    true_label = y.item()
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    print("Test index:", TEST_INDEX)
    print("True label:", true_label)
    print("Predicted label:", pred)
    print("Probabilities:", probs)

    loss = F.cross_entropy(logits, y)
    print("Test loss:", loss.item())

    flat_grad = grad_of_loss(model, x, y)

    print("Weight shape:", tuple(model.linear.weight.shape))
    print("Flattened gradient shape:", tuple(flat_grad.shape))
    print("Expected parameter count:", 28 * 28 * 10)

    img = x[0, 0].detach().cpu().numpy()
    plt.imshow(img, cmap="gray")
    plt.title(f"index={TEST_INDEX}, true={true_label}, pred={pred}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/test_point_inspect.png", dpi=150)
    print("Saved image to outputs/test_point_inspect.png")


if __name__ == "__main__":
    main()
