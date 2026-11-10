"""Training script (skeleton — dataset wiring still in progress)."""

import argparse

import torch
import torch.nn as nn

from model import MultimodalDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/FakeAVCeleb_v1.2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalDetector().to(device)

    fake_loss = nn.CrossEntropyLoss()  # real / fake
    cat_loss = nn.CrossEntropyLoss()   # which modality was faked
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    # TODO: cosine lr schedule + early stopping

    # TODO: plug in the data loader once frame extraction is done
    # for epoch in range(args.epochs):
    #     for frames, spec, y, c in loader:
    #         out_fake, out_cat = model(frames.to(device), spec.to(device))
    #         loss = fake_loss(out_fake, y) + cat_loss(out_cat, c)
    #         optim.zero_grad(); loss.backward(); optim.step()
    print("data loader not wired yet — nothing to train on for now")


if __name__ == "__main__":
    main()
