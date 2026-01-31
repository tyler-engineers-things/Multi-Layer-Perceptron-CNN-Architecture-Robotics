import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt

DATASETS_ROOT = os.path.join(os.path.dirname(__file__), "training_dataset/bin_pushing/")

all_trajs = sorted([
    name for name in os.listdir(DATASETS_ROOT)
    if os.path.isdir(os.path.join(DATASETS_ROOT, name))
])

print("All trajectory folders:", all_trajs)

train_traj_names = np.random.choice(all_trajs, size=24, replace=False)
val_traj_names = [all_trajs[i] for i in range(len(all_trajs)) if all_trajs[i] not in train_traj_names]


class MultiBowlPushDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, allowed_traj_names=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.allowed_traj_names = allowed_traj_names

        self.datasets = []
        self.index_map = []

        for name in sorted(os.listdir(self.root)):
            if self.allowed_traj_names is not None and name not in self.allowed_traj_names:
                continue

            dpath = os.path.join(self.root, name)
            if not os.path.isdir(dpath):
                continue

            if (os.path.isdir(os.path.join(dpath, "rs_rgb")) and os.path.isdir(os.path.join(dpath, "rs_depth")) and os.path.isdir(os.path.join(dpath, "logi_rgb")) and os.path.exists(os.path.join(dpath, "joint_positions.npy"))):
                joints = np.load(os.path.join(dpath, "joint_positions.npy")).astype(np.float32)
                ds_idx = len(self.datasets)
                self.datasets.append({"root": dpath, "joints": joints})
                for t in range(joints.shape[0]):
                    self.index_map.append((ds_idx, t))

        if len(self.datasets) == 0:
            raise RuntimeError(f"No datasets found under: {self.root}")

        print(f"Loaded {len(self.datasets)} datasets, total {len(self.index_map)} samples.")

    def __len__(self):
        return len(self.index_map)

    def _img_path(self, droot, subdir, idx):
        return os.path.join(droot, subdir, f"{idx:06d}.png")

    def __getitem__(self, idx):
        ds_idx, t = self.index_map[idx]
        entry = self.datasets[ds_idx]
        droot = entry["root"]

        joints = entry["joints"][t]

        top_rgb = read_image(self._img_path(droot, "rs_rgb", t)).float() / 255.0
        top_depth = read_image(self._img_path(droot, "rs_depth", t)).float() / 255.0
        side_rgb = read_image(self._img_path(droot, "logi_rgb", t)).float() / 255.0

        top_rgb = top_rgb[:3, :, :]
        top_depth = top_depth[:1, :, :]
        side_rgb = side_rgb[:3, :, :]

        return {
            "top_rgb": top_rgb,
            "top_depth": top_depth,
            "side_rgb": side_rgb,
            "joint_positions": torch.tensor(joints, dtype=torch.float32),
        }



class BowlPushPolicy(nn.Module):
    def __init__(self, num_actions=7, state_dim=7):
        super().__init__()

        img_channels = 7

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4 + state_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, img, state):
        x = torch.flatten(self.avgpool(self.conv(img)), 1)
        x = torch.cat([x, state], dim=1)
        return self.mlp(x)


def train_policy(data_root, save_path, epochs, batch_size, lr, val_split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_trajs = sorted([
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    ])

    if len(all_trajs) == 0:
        raise RuntimeError(f"No trajectory folders found under: {data_root}")

    num_val_traj = max(1, int(len(all_trajs) * val_split))
    num_val_traj = min(num_val_traj, len(all_trajs) - 1) if len(all_trajs) > 1 else 1

    rng = np.random.default_rng(42)
    val_indices = rng.choice(len(all_trajs), size=num_val_traj, replace=False)

    val_traj_names = [all_trajs[i] for i in val_indices]
    train_traj_names = [name for i, name in enumerate(all_trajs) if i not in val_indices]

    print("Trajectory folders (all):", all_trajs)
    print("Train trajectories:", train_traj_names)
    print("Val trajectories:  ", val_traj_names)

    train_dataset = MultiBowlPushDataset(data_root, allowed_traj_names=train_traj_names)
    val_dataset = MultiBowlPushDataset(data_root, allowed_traj_names=val_traj_names)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    policy = BowlPushPolicy().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        running_loss = 0.0

        policy.train()
        for batch_i, batch in enumerate(train_loader):
            top_rgb = batch["top_rgb"].to(device)
            top_depth = batch["top_depth"].to(device)
            side_rgb = batch["side_rgb"].to(device)
            joints = batch["joint_positions"].to(device)

            img = torch.cat([top_rgb, top_depth, side_rgb], dim=1)
            state = joints.clone()

            pred = policy(img, state)
            loss = loss_fn(pred, joints)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_loss = running_loss / len(train_loader)
        train_losses.append(train_epoch_loss)

        policy.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                top_rgb = batch["top_rgb"].to(device)
                top_depth = batch["top_depth"].to(device)
                side_rgb = batch["side_rgb"].to(device)
                joints = batch["joint_positions"].to(device)

                img = torch.cat([top_rgb, top_depth, side_rgb], dim=1)
                state = joints.clone()

                pred = policy(img, state)
                loss = loss_fn(pred, joints)
                val_loss_total += loss.item()

        val_epoch_loss = val_loss_total / len(val_loader)
        val_losses.append(val_epoch_loss)

        print(f"  Train Loss: {train_epoch_loss:.6f}")
        print(f"  Val Loss:   {val_epoch_loss:.6f}")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save({"model": policy.state_dict()}, save_path)
            print(f"  → Saved new best checkpoint, val loss {best_val_loss:.6f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")

    epochs_axis = list(range(1, epochs + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_axis, train_losses, label="Training Loss", marker='o')
    plt.plot(epochs_axis, val_losses,   label="Validation Loss", marker='o')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.savefig("loss_curve_real_bowl.png")
    print("Saved loss plot as loss_curve_real_bowl.png")
    try:
        plt.show()
    except:
        pass


if __name__ == "__main__":
    train_policy(
        data_root=DATASETS_ROOT,
        save_path="real_bowl_policy_checkpoint.pt",
        epochs=10,
        batch_size=32,
        lr=1e-4,
        val_split=0.2
    )