import torch
import decord
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader
from typing import Callable, Optional, Any
from .traj_dset import TrajDataset, get_train_val_sliced
decord.bridge.set_bridge("torch")


class PointMazeDataset(TrajDataset):
    """
    Point Maze Dataset with optional Domain Randomization support.
    
    When domain randomization is enabled (use_dr=True), this dataset loads
    from a DR-processed dataset that contains re-rendered observations with
    varied backgrounds and distractors.
    
    Args:
        data_path: Path to the dataset directory
        n_rollout: Number of rollouts to load (None for all)
        transform: Image transform to apply
        normalize_action: Whether to normalize actions
        action_scale: Scale factor for actions
        use_dr: Whether to use domain randomization dataset
        dr_data_path: Optional separate path for DR dataset (if different from data_path)
    """
    def __init__(
        self,
        data_path: str = "data/point_maze",
        n_rollout: Optional[int] = None,
        transform: Optional[Callable] = None,
        normalize_action: bool = False,
        action_scale: float = 1.0,
        use_dr: bool = False,
        dr_data_path: Optional[str] = None,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.normalize_action = normalize_action
        self.use_dr = use_dr
        
        # Determine which path to use for observations
        if use_dr and dr_data_path is not None:
            self.obs_data_path = Path(dr_data_path)
            print(f"Domain Randomization enabled: loading observations from {self.obs_data_path}")
        else:
            self.obs_data_path = self.data_path
            if use_dr:
                print(f"Domain Randomization enabled: using same path for DR data")
        
        # Load DR metadata if available
        self.dr_metadata = None
        dr_metadata_path = self.obs_data_path / "dr_metadata.pth"
        if use_dr and dr_metadata_path.exists():
            self.dr_metadata = torch.load(dr_metadata_path)
            n_dr = len(self.dr_metadata.get('dr_trajectory_indices', []))
            print(f"  DR metadata loaded: {n_dr} trajectories with domain randomization")
        
        # Load trajectory data (always from original data_path)
        states = torch.load(self.data_path / "states.pth").float()
        self.states = states
        self.actions = torch.load(self.data_path / "actions.pth").float()
        self.actions = self.actions / action_scale  # scaled back up in env
        self.seq_lengths = torch.load(self.data_path / 'seq_lengths.pth')

        self.n_rollout = n_rollout
        if self.n_rollout:
            n = self.n_rollout
        else:
            n = len(self.states)

        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]
        self.proprios = self.states.clone()
        print(f"Loaded {n} rollouts")

        self.action_dim = self.actions.shape[-1]
        self.state_dim = self.states.shape[-1]
        self.proprio_dim = self.proprios.shape[-1]

        if normalize_action:
            self.action_mean, self.action_std = self.get_data_mean_std(self.actions, self.seq_lengths)
            self.state_mean, self.state_std = self.get_data_mean_std(self.states, self.seq_lengths)
            self.proprio_mean, self.proprio_std = self.get_data_mean_std(self.proprios, self.seq_lengths)
        else:
            self.action_mean = torch.zeros(self.action_dim)
            self.action_std = torch.ones(self.action_dim)
            self.state_mean = torch.zeros(self.state_dim)
            self.state_std = torch.ones(self.state_dim)
            self.proprio_mean = torch.zeros(self.proprio_dim)
            self.proprio_std = torch.ones(self.proprio_dim)

        self.actions = (self.actions - self.action_mean) / self.action_std
        self.proprios = (self.proprios - self.proprio_mean) / self.proprio_std
    
    def get_data_mean_std(self, data, traj_lengths):
        all_data = []
        for traj in range(len(traj_lengths)):
            traj_len = traj_lengths[traj]
            traj_data = data[traj, :traj_len]
            all_data.append(traj_data)
        all_data = torch.vstack(all_data)
        data_mean = torch.mean(all_data, dim=0)
        data_std = torch.std(all_data, dim=0)
        return data_mean, data_std

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        # Load observations from obs_data_path (may be DR path or original path)
        obs_dir = self.obs_data_path / "obses"
        image = torch.load(obs_dir / f"episode_{idx:03d}.pth")
        proprio = self.proprios[idx, frames]
        act = self.actions[idx, frames]
        state = self.states[idx, frames]

        image = image[frames]  # THWC
        image = image / 255.0
        image = rearrange(image, "T H W C -> T C H W")
        if self.transform:
            image = self.transform(image)
        obs = {
            "visual": image,
            "proprio": proprio
        }
        return obs, act, state, {} # env_info

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

    def preprocess_imgs(self, imgs):
        if isinstance(imgs, np.ndarray):
            raise NotImplementedError
        elif isinstance(imgs, torch.Tensor):
            return rearrange(imgs, "b h w c -> b c h w") / 255.0
    
    def is_dr_trajectory(self, idx: int) -> bool:
        """
        Check if a trajectory has domain randomization applied.
        
        Args:
            idx: Trajectory index
            
        Returns:
            True if the trajectory has DR, False otherwise
        """
        if not self.use_dr or self.dr_metadata is None:
            return False
        dr_indices = self.dr_metadata.get('dr_trajectory_indices', [])
        return idx in dr_indices
    
    def get_dr_config(self, idx: int) -> Optional[dict]:
        """
        Get the DR configuration for a trajectory (if available).
        
        Args:
            idx: Trajectory index
            
        Returns:
            Dict with chunk configs, or None if not a DR trajectory
        """
        if not self.is_dr_trajectory(idx):
            return None
        return self.dr_metadata.get('trajectory_configs', {}).get(idx, None)
        
def load_point_maze_slice_train_val(
    transform,
    n_rollout: int = 50,
    data_path: str = 'data/pusht_dataset',
    normalize_action: bool = False,
    split_ratio: float = 0.8,
    num_hist: int = 0,
    num_pred: int = 0,
    frameskip: int = 0,
    use_dr: bool = False,
    dr_data_path: Optional[str] = None,
):
    """
    Load point maze dataset with train/val split and optional domain randomization.
    
    Args:
        transform: Image transform to apply
        n_rollout: Number of rollouts to load
        data_path: Path to the dataset
        normalize_action: Whether to normalize actions
        split_ratio: Train/val split ratio
        num_hist: Number of history frames
        num_pred: Number of prediction frames
        frameskip: Frame skip for slicing
        use_dr: Whether to use domain randomization dataset (only for has_bisim=False)
        dr_data_path: Optional separate path for DR dataset
        
    Returns:
        Tuple of (datasets dict, traj_dset dict)
    """
    dset = PointMazeDataset(
        n_rollout=n_rollout,
        transform=transform,
        data_path=data_path,
        normalize_action=normalize_action,
        use_dr=use_dr,
        dr_data_path=dr_data_path,
    )
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset, 
        train_fraction=split_ratio, 
        num_frames=num_hist + num_pred, 
        frameskip=frameskip
    )

    datasets = {}
    datasets['train'] = train_slices
    datasets['valid'] = val_slices
    traj_dset = {}
    traj_dset['train'] = dset_train
    traj_dset['valid'] = dset_val
    return datasets, traj_dset
