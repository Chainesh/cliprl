"""
CLIP-RL Training Loop

Trains the CLIP model on (instruction, trajectory) pairs from all base tasks.

Because N (number of base tasks) is small (~8), the entire dataset fits in
one batch. We run many epochs over this single batch.

Key metrics tracked:
    - InfoNCE loss (should decrease)
    - Diagonal similarity (positive pairs, should increase)
    - Off-diagonal similarity (negative pairs, should decrease)
    - Temperature (should stabilize)
    - Alignment accuracy: what % of tasks does CLIP correctly identify
      as most similar to their own instruction? (should → 100%)
"""

import torch
import torch.optim as optim
import numpy as np
import os, sys, time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_rl_model.dataset import CLIPDataset
from clip_rl_model.model   import CLIPModel
from utils.config import CLIP_TRAIN, CHECKPOINTS, RESULTS


def train_clip(
    device:   str = "cpu",
    epochs:   int = CLIP_TRAIN["epochs"],
    lr:       float = CLIP_TRAIN["lr"],
    save_every: int = 20,
    verbose:  bool = True,
) -> CLIPModel:
    """
    Full CLIP training loop.

    Args:
        device:     "cpu" or "cuda"
        epochs:     number of epochs over the full dataset
        lr:         learning rate for Adam
        save_every: checkpoint every N epochs
        verbose:    print progress

    Returns:
        Trained CLIPModel
    """

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = CLIPDataset(device=device).load()
    N = len(dataset)

    if N < 2:
        raise RuntimeError(f"Need at least 2 tasks for CLIP training, got {N}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = CLIPModel(device=device)
    print(f"  CLIP model trainable params: {model.trainable_param_count():,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Only train the projection heads and temperature — NOT the SBERT backbone
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr  = lr,
        eps = 1e-8,
    )

    # Cosine annealing: smoothly reduce LR over training
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Unpack dataset ────────────────────────────────────────────────────────
    instructions = dataset.instructions
    traj_tensors = dataset.all_traj_tensors
    traj_lengths = dataset.all_traj_lengths

    # ── Metrics history ───────────────────────────────────────────────────────
    history = {
        "loss"          : [],
        "diag_sim"      : [],   # avg similarity of positive pairs
        "offdiag_sim"   : [],   # avg similarity of negative pairs
        "temperature"   : [],
        "accuracy"      : [],   # % of tasks correctly matched
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  CLIP Training  |  N={N} tasks  |  epochs={epochs}  |  device={device}")
        print(f"{'='*60}\n")
        print(f"  {'Epoch':>6}  {'Loss':>8}  {'DiagSim':>9}  {'OffDiag':>9}  {'Temp':>7}  {'Acc':>6}")
        print(f"  {'-'*55}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        out  = model(instructions, traj_tensors, traj_lengths)
        loss = out["loss"]

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ── Diagnostics ───────────────────────────────────────────────────────
        with torch.no_grad():
            S    = out["similarity_matrix"]   # (N, N) — already divided by temp
            temp = out["temperature"]

            # Similarity values (before temperature scaling, i.e. actual cosine sim)
            lang_e = out["lang_embs"]
            traj_e = out["traj_embs"]
            raw_S  = torch.matmul(lang_e, traj_e.T)  # (N, N) cosine similarities

            diag_sim    = raw_S.diag().mean().item()
            mask_off    = ~torch.eye(N, dtype=torch.bool, device=raw_S.device)
            offdiag_sim = raw_S[mask_off].mean().item()

            # Accuracy: for each instruction, does the top-1 trajectory match?
            preds    = S.argmax(dim=1)
            labels   = torch.arange(N, device=S.device)
            accuracy = (preds == labels).float().mean().item()

        history["loss"].append(loss.item())
        history["diag_sim"].append(diag_sim)
        history["offdiag_sim"].append(offdiag_sim)
        history["temperature"].append(temp)
        history["accuracy"].append(accuracy)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"  {epoch:>6}  {loss.item():>8.4f}  "
                f"{diag_sim:>9.4f}  {offdiag_sim:>9.4f}  "
                f"{temp:>7.4f}  {accuracy*100:>5.1f}%"
            )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % save_every == 0:
            ckpt_path = os.path.join(CHECKPOINTS, f"clip_epoch_{epoch:04d}.pt")
            model.save(ckpt_path)

    elapsed = time.time() - start_time

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(CHECKPOINTS, "clip_model_final.pt")
    model.save(final_path)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training complete in {elapsed:.1f}s")
        print(f"  Final loss:          {history['loss'][-1]:.4f}")
        print(f"  Final diag sim:      {history['diag_sim'][-1]:.4f}")
        print(f"  Final off-diag sim:  {history['offdiag_sim'][-1]:.4f}")
        print(f"  Final temperature:   {history['temperature'][-1]:.4f}")
        print(f"  Final accuracy:      {history['accuracy'][-1]*100:.1f}%")
        print(f"{'='*60}\n")

    # ── Plot training curves ───────────────────────────────────────────────────
    _plot_training_curves(history)

    return model


def _plot_training_curves(history: dict):
    """Save training diagnostic plots to results/."""
    os.makedirs(RESULTS, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("CLIP-RL Training Curves", fontsize=14)

    axes[0, 0].plot(history["loss"])
    axes[0, 0].set_title("InfoNCE Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)

    axes[0, 1].plot(history["diag_sim"],   label="Positive pairs (diagonal)")
    axes[0, 1].plot(history["offdiag_sim"], label="Negative pairs (off-diagonal)")
    axes[0, 1].set_title("Cosine Similarity")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(history["temperature"])
    axes[1, 0].set_title("Temperature (δ)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True)

    axes[1, 1].plot([a * 100 for a in history["accuracy"]])
    axes[1, 1].set_title("Alignment Accuracy (%)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].grid(True)

    plt.tight_layout()
    out_path = os.path.join(RESULTS, "clip_training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {out_path}")


def load_trained_clip(device: str = "cpu") -> CLIPModel:
    """Load the final trained CLIP model."""
    model = CLIPModel(device=device)
    path  = os.path.join(CHECKPOINTS, "clip_model_final.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained CLIP model found at {path}. Run phase 3 first.")
    model.load(path)
    model.eval()
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_clip(device=device)