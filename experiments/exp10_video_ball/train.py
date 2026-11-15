"""Step 2: train the ball heat-map net on the crops of make_dataset.py.

    python experiments/exp10_video_ball/train.py [--epochs 40] [--device auto]

Uses a GPU when there is one (minutes), otherwise the CPU (hours for the full set; fine for a
test with --epochs 1). Writes models/ballnet.pt and models/ballnet.json with the validation
numbers and the score threshold: the score above which 1% of ball-free validation crops fire.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

import vb

SIGMA = 2.0            # px, width of the target blob
HIT_PX = 4.0           # a detection within this distance of the target is correct


def targets(labels, device):
    """(B, 1, CROP, CROP) Gaussian blobs at the labels; all zero for negatives (label -1)."""
    grid = torch.arange(vb.CROP, device=device, dtype=torch.float32)
    dx = grid[None, None, :] - labels[:, 0, None, None]
    dy = grid[None, :, None] - labels[:, 1, None, None]
    heat = torch.exp(-(dx ** 2 + dy ** 2) / (2 * SIGMA ** 2))
    return (heat * (labels[:, 0] >= 0)[:, None, None]).unsqueeze(1)


def batch_of(x, labels, index, device, augment, rng):
    index = np.sort(index)
    xb = torch.from_numpy(x[index]).to(device).float() / 255.0
    lb = torch.from_numpy(labels[index]).to(device)
    if augment:
        flip = torch.from_numpy(rng.random(len(index)) < 0.5).to(device)
        xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
        lb = lb.clone()
        lb[:, 0] = torch.where(flip & (lb[:, 0] >= 0), vb.CROP - 1 - lb[:, 0], lb[:, 0])
        gain = torch.from_numpy(rng.uniform(0.8, 1.2, (len(index), 1, 1, 1)).astype(np.float32)).to(device)
        xb = (xb * gain).clamp(0, 1)
    return xb, lb


@torch.no_grad()
def evaluate(net, x, labels, index, device, batch=128):
    """Scores of the ball-free crops, then score and "right place" of the crops with a ball."""
    net.eval()
    index = np.sort(index)
    scores, correct = [], []
    for start in range(0, len(index), batch):
        xb, lb = batch_of(x, labels, index[start: start + batch], device, False, None)
        heat = torch.sigmoid(net(xb)).flatten(1)
        best, where = heat.max(dim=1)
        px, py = (where % vb.CROP).float(), (where // vb.CROP).float()
        scores.append(best.cpu().numpy())
        correct.append((torch.hypot(px - lb[:, 0], py - lb[:, 1]) <= HIT_PX).cpu().numpy())
    scores, correct = np.concatenate(scores), np.concatenate(correct)
    positive = labels[index, 0] >= 0
    return scores[~positive], scores[positive], correct[positive]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = vb.pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = np.load(os.path.join(vb.CACHE, "dataset.npz"))
    x, labels, val = data["x"], data["label"], data["val"]
    train_index, val_index = np.flatnonzero(~val), np.flatnonzero(val)
    print("device {}; {} training crops, {} validation crops".format(
        device, len(train_index), len(val_index)), flush=True)

    net = vb.build_net().to(device)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    per_epoch = max(1, len(train_index) // args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=args.epochs * per_epoch)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best, started = None, time.time()
    os.makedirs(vb.MODELS, exist_ok=True)
    for epoch in range(args.epochs):
        net.train()
        order = rng.permutation(train_index)
        total = 0.0
        for step in range(per_epoch):
            xb, lb = batch_of(x, labels, order[step * args.batch: (step + 1) * args.batch], device, True, rng)
            target = targets(lb, device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = net(xb)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits.float(), target, weight=1.0 + 60.0 * target)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            schedule.step()
            total += loss.item()
        negative, positive, correct = evaluate(net, x, labels, val_index, device)
        threshold = float(np.quantile(negative, 0.99)) if len(negative) else 0.5
        recall = float(np.mean(correct & (positive >= threshold))) if len(positive) else 0.0
        located = float(np.mean(correct)) if len(positive) else 0.0
        print("epoch {:>2}  loss {:.4f}  validation: ball located {:.1%}, found above the threshold {:.1%} "
              "(threshold {:.2f})  [{:.0f}s]".format(epoch + 1, total / per_epoch, located, recall, threshold,
                                                      time.time() - started), flush=True)
        if best is None or recall >= best["recall"]:
            best = {"recall": recall, "located": located, "threshold": threshold, "epoch": epoch + 1,
                    "train_crops": int(len(train_index)), "val_crops": int(len(val_index))}
            torch.save(net.state_dict(), os.path.join(vb.MODELS, "ballnet.pt"))
            with open(os.path.join(vb.MODELS, "ballnet.json"), "w") as handle:
                json.dump(best, handle, indent=2)
    print("best epoch {epoch}: found {recall:.1%} of the validation balls at a 1% false-alarm rate per crop "
          "(threshold {threshold:.2f})".format(**best))


if __name__ == "__main__":
    main()
