import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt

print("Absolute version")

DATASETS_ROOT = os.path.join(
    os.path.dirname(__file__),
    "training_dataset",
    "pick_and_place",
)

GRIPPER_OPEN_THRESHOLD = 0.02

class MultiPickPlaceDataset(Dataset):

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
            top_rgb_dir = os.path.join(dpath, "rs_rgb")
            top_depth_dir = os.path.join(dpath, "rs_depth")
            side_rgb_dir = os.path.join(dpath, "logi_rgb")
            joints_path = os.path.join(dpath, "joint_positions.npy")
            if (os.path.isdir(top_rgb_dir) and os.path.isdir(top_depth_dir) and os.path.isdir(side_rgb_dir) and os.path.exists(joints_path)):
                joints = np.load(joints_path).astype(np.float32)
                if joints.ndim != 2 or joints.shape[1] < 7:
                    raise ValueError(f"Expected joints shape [T, 7], got {joints.shape} in {joints_path}")

                ds_idx = len(self.datasets)
                self.datasets.append({"root": dpath, "joints": joints})

                num_steps = joints.shape[0]
                for t in range(num_steps):
                    self.index_map.append((ds_idx, t))

        if len(self.datasets) == 0:
            raise RuntimeError(f"No valid teleoperation datasets found under root: {self.root}")

        print(
            f"Loaded {len(self.datasets)} trajectory folder(s), "
            f"total {len(self.index_map)} samples from {self.root}."
        )

    def __len__(self):
        return len(self.index_map)

    def _img_path(self, droot: str, subdir: str, idx: int) -> str:
        return os.path.join(droot, subdir, f"{idx:06d}.png")

    def __getitem__(self, idx):
        ds_idx, t = self.index_map[idx]
        entry = self.datasets[ds_idx]
        droot = entry["root"]

        joints_t = entry["joints"][t]

        if t == 0:
            prev_joints = joints_t
        else:
            prev_joints = entry["joints"][t - 1]

        top_rgb = read_image(self._img_path(droot, "rs_rgb", t)).float() / 255.0
        top_depth = read_image(self._img_path(droot, "rs_depth", t)).float() / 255.0
        side_rgb = read_image(self._img_path(droot, "logi_rgb", t)).float() / 255.0

        top_rgb = top_rgb[:3, :, :]
        top_depth = top_depth[:1, :, :]
        side_rgb = side_rgb[:3, :, :]

        sample = {
            "top_rgb": top_rgb,
            "top_depth": top_depth,
            "side_rgb": side_rgb,
            "joint_positions": torch.tensor(joints_t, dtype=torch.float32),
            "prev_joint_positions": torch.tensor(prev_joints, dtype=torch.float32),
        }

        if self.transform is not None:
            sample["top_rgb"] = self.transform(sample["top_rgb"])
            sample["top_depth"] = self.transform(sample["top_depth"])
            sample["side_rgb"] = self.transform(sample["side_rgb"])

        if self.target_transform is not None:
            sample["joint_positions"] = self.target_transform(sample["joint_positions"])
            sample["prev_joint_positions"] = self.target_transform(
                sample["prev_joint_positions"]
            )

        return sample


class PickPlacePolicy(nn.Module):
    def __init__(self, img_channels: int = 7, state_dim: int = 7, hidden_dim: int = 256):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        self.fusion_mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4 + state_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.joint_head = nn.Linear(hidden_dim, 6)

        self.gripper_head = nn.Linear(hidden_dim, 1)

        nn.init.uniform_(self.joint_head.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.joint_head.bias, -1e-3, 1e-3)
        nn.init.uniform_(self.gripper_head.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.gripper_head.bias, -1e-3, 1e-3)

    def forward(self, img: torch.Tensor, state: torch.Tensor):
        x = torch.flatten(self.avgpool(self.conv(img)), 1)
        x = torch.cat([x, state], dim=1)
        x = self.fusion_mlp(x)

        joint_angles = self.joint_head(x)
        gripper_logit = self.gripper_head(x)
        gripper_prob = torch.sigmoid(gripper_logit)

        return joint_angles, gripper_prob

def train_pick_place_policy(data_root=DATASETS_ROOT, save_path="real_pick_place_policy_checkpoint.pt", epochs=10, batch_size=16, lr=1e-4, val_split=0.2, num_joints=6, gripper_index=6):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_trajs = sorted([
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    ])

    if not all_trajs:
        raise RuntimeError(f"No trajectory folders found in {data_root}")

    indices = np.arange(len(all_trajs))
    np.random.shuffle(indices)

    num_train = max(1, int(len(all_trajs) * (1.0 - val_split)))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_traj_names = [all_trajs[i] for i in train_indices]
    val_traj_names = [all_trajs[i] for i in val_indices] if len(val_indices) > 0 else []

    print("All trajectories:", all_trajs)
    print("Train trajs:", train_traj_names)
    print("Val trajs:  ", val_traj_names)

    train_dataset = MultiPickPlaceDataset(data_root, allowed_traj_names=train_traj_names)
    val_dataset = MultiPickPlaceDataset(data_root, allowed_traj_names=val_traj_names)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    state_dim = train_dataset.datasets[0]["joints"].shape[1]
    policy = PickPlacePolicy(img_channels=7, state_dim=state_dim, hidden_dim=256).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    joint_loss_fn = nn.L1Loss()
    gripper_loss_fn = nn.BCELoss()

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        policy.train()
        running_loss = 0.0

        for batch in train_loader:
            top_rgb = batch["top_rgb"].to(device)
            top_depth = batch["top_depth"].to(device)
            side_rgb = batch["side_rgb"].to(device)

            joints = batch["joint_positions"].to(device)
            prev_joints = batch["prev_joint_positions"].to(device)

            img = torch.cat([top_rgb, top_depth, side_rgb], dim=1)

            state = prev_joints

            target_joints = joints[:, :num_joints]

            gripper_joint = joints[:, gripper_index]
            gripper_target = (gripper_joint > GRIPPER_OPEN_THRESHOLD).float().unsqueeze(1)

            joint_pred, gripper_prob = policy(img, state)

            joint_loss = joint_loss_fn(joint_pred, target_joints)
            gripper_loss = gripper_loss_fn(gripper_prob, gripper_target)
            loss = joint_loss + 0.5 * gripper_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
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
                prev_joints = batch["prev_joint_positions"].to(device)

                img = torch.cat([top_rgb, top_depth, side_rgb], dim=1)
                state = prev_joints

                target_joints = joints[:, :num_joints]

                gripper_joint = joints[:, gripper_index]
                gripper_target = (gripper_joint > GRIPPER_OPEN_THRESHOLD).float().unsqueeze(1)

                joint_pred, gripper_prob = policy(img, state)

                joint_loss = joint_loss_fn(joint_pred, target_joints)
                gripper_loss = gripper_loss_fn(gripper_prob, gripper_target)
                loss = joint_loss + 0.5 * gripper_loss

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
    plt.plot(epochs_axis, train_losses, label="Training Loss", marker="o")
    plt.plot(epochs_axis, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss (Pick-and-Place Policy)")
    plt.grid(True)
    plt.legend()
    plt.savefig("real_pick_place_loss_curve.png") # Change based on model trajectory graph desired
    print("Saved loss plot as real_pick_place_loss_curve.png")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    train_pick_place_policy(
        data_root=DATASETS_ROOT,
        save_path="real_pick_place_policy_checkpoint.pt",
        epochs=15,
        batch_size=16,
        lr=1e-4,
        val_split=0.2,
        num_joints=6,
        gripper_index=6
    )
