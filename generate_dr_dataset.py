#!/usr/bin/env python3

import argparse
import shutil
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

from env.pointmaze.point_maze_wrapper import PointMazeWrapper
from datasets.domain_randomization import (
    CANONICAL_BACKGROUND,
    DR_BACKGROUNDS,
    DISTRACTOR_Z,
    sample_distractor_xy,
    sample_distractor_rgba,
    sample_dr_background,
)


U_MAZE = "#####\\#GOO#\\###O#\\#OOO#\\#####"


def create_env_with_background(
    background_config: Dict[str, str],
    maze_spec: str = U_MAZE,
    render_size: int = 224,
) -> PointMazeWrapper:
    """Create a PointMazeWrapper with a specific background configuration."""
    # Pass background as a dict so PointMazeWrapper uses it directly
    # (instead of defaulting to 'default' and overwriting our config)
    env = PointMazeWrapper(
        maze_spec=maze_spec,
        return_value="obs",
        background=background_config,  # Pass as dict, not separate kwargs
        with_target=False,
        reset_target=False,
    )
    env.prepare_for_render()
    
    # Move target marker off-screen since with_target=False
    # This must be called after prepare_for_render() to ensure rendering context exists
    env.set_marker()
    
    return env


def render_frame_with_state(
    env: PointMazeWrapper,
    state: np.ndarray,
    render_size: int = 224,
    distractor_config: Optional[Dict] = None,
) -> np.ndarray:
    """
    Render a single frame by setting the environment to a specific state.
    
    Args:
        env: The PointMazeWrapper environment
        state: The state to render (qpos + qvel, shape: (4,))
        render_size: Size of the rendered image
        distractor_config: Optional dict with 'pos' (x, y) and 'rgba' (r, g, b, a)
        
    Returns:
        Rendered image as numpy array (H, W, C)
    """
    # set the state
    qpos = state[:2]
    qvel = state[2:4] if len(state) >= 4 else np.array([0.0, 0.0])
    env.set_state(qpos, qvel)
    
    # Ensure target marker stays off-screen (with_target=False)
    env.set_marker()
    
    # set distractor if provided
    if distractor_config is not None:
        distractor_site_id = env.model.site_name2id('distractor_site')
        x, y = distractor_config['pos']
        env.sim.data.site_xpos[distractor_site_id] = np.array([x, y, DISTRACTOR_Z])
        r, g, b, a = distractor_config['rgba']
        env.model.site_rgba[distractor_site_id] = np.array([r, g, b, a])
    else:
        # hide distractor by making it transparent
        distractor_site_id = env.model.site_name2id('distractor_site')
        env.model.site_rgba[distractor_site_id] = np.array([0, 0, 0, 0])
    
    # render
    img = env.sim.render(render_size, render_size)
    return img


def render_trajectory_with_dr(
    states: np.ndarray,
    seq_length: int,
    rng: np.random.Generator,
    render_size: int = 224,
    maze_spec: str = U_MAZE,
) -> Tuple[np.ndarray, Dict]:
    """
    Re-render a trajectory with domain randomization.
    
    The entire trajectory uses the same (background, distractor) for consistency.
    Distractor is included 50% of the time.
    
    Args:
        states: All states for this trajectory (T, state_dim)
        seq_length: Actual length of this trajectory
        rng: NumPy random generator for reproducibility
        render_size: Size of rendered images
        maze_spec: Maze specification string
        
    Returns:
        Tuple of (rendered_images, trajectory_config)
        - rendered_images: (T, H, W, C) numpy array
        - trajectory_config: Dict with background and distractor config for the trajectory
    """
    # Sample background for the entire trajectory
    bg_config = sample_dr_background(rng)
    
    # Randomly decide whether to include distractor (50% chance)
    has_distractor = rng.random() < 0.5
    
    if has_distractor:
        distractor_pos = sample_distractor_xy(rng)
        distractor_rgba = sample_distractor_rgba(rng)
        distractor_config = {
            'pos': distractor_pos,
            'rgba': distractor_rgba,
        }
    else:
        distractor_config = None
    
    trajectory_config = {
        'background': bg_config.copy(),
        'distractor': distractor_config.copy() if distractor_config is not None else None,
        'has_distractor': has_distractor,
    }
    
    # Create environment with this background
    env = create_env_with_background(bg_config, maze_spec, render_size)
    
    # Render all frames with the same background/distractor
    rendered_images = []
    for frame_idx in range(seq_length):
        state = states[frame_idx]
        img = render_frame_with_state(
            env, state, render_size, distractor_config
        )
        rendered_images.append(img)
    
    env.close()
    
    rendered_images = np.stack(rendered_images, axis=0)
    return rendered_images, trajectory_config


def render_trajectory_canonical(
    states: np.ndarray,
    seq_length: int,
    render_size: int = 224,
    maze_spec: str = U_MAZE,
) -> np.ndarray:
    """
    Re-render a trajectory with the canonical background (no distractor).
    
    Args:
        states: All states for this trajectory (T, state_dim)
        seq_length: Actual length of this trajectory
        render_size: Size of rendered images
        maze_spec: Maze specification string
        
    Returns:
        rendered_images: (T, H, W, C) numpy array
    """
    rendered_images = []
    
    # create environment with canonical background
    env = create_env_with_background(CANONICAL_BACKGROUND, maze_spec, render_size)
    
    # render each frame
    for frame_idx in range(seq_length):
        state = states[frame_idx]
        img = render_frame_with_state(
            env, state, render_size, distractor_config=None
        )
        rendered_images.append(img)
    
    env.close()
    
    rendered_images = np.stack(rendered_images, axis=0)
    return rendered_images


def main():
    parser = argparse.ArgumentParser(
        description='Generate domain-randomized point maze dataset'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to the original point maze dataset'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Path to save the DR dataset (can be same as data-path to overwrite)'
    )
    parser.add_argument(
        '--dr-fraction',
        type=float,
        default=0.4,
        help='Fraction of trajectories to apply DR to (default: 0.4)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='DEPRECATED: Background is now consistent for entire trajectory. This parameter is ignored.'
    )
    parser.add_argument(
        '--render-size',
        type=int,
        default=224,
        help='Size of rendered images (default: 224)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--maze-spec',
        type=str,
        default='U_MAZE',
        choices=['U_MAZE'],
        help='Maze specification (default: U_MAZE)'
    )
    parser.add_argument(
        '--re-render-canonical',
        action='store_true',
        help='Also re-render canonical trajectories (useful if original data has different background)'
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    
    # load existing dataset
    print(f"Loading dataset from {data_path}...")
    states = torch.load(data_path / "states.pth")
    actions = torch.load(data_path / "actions.pth")
    seq_lengths = torch.load(data_path / "seq_lengths.pth")
    
    n_trajectories = len(seq_lengths)
    print(f"Found {n_trajectories} trajectories")
    
    # create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    obses_path = output_path / "obses"
    obses_path.mkdir(parents=True, exist_ok=True)
    
    # copy non-visual data
    if output_path != data_path:
        print("Copying states.pth, actions.pth, seq_lengths.pth...")
        shutil.copy(data_path / "states.pth", output_path / "states.pth")
        shutil.copy(data_path / "actions.pth", output_path / "actions.pth")
        shutil.copy(data_path / "seq_lengths.pth", output_path / "seq_lengths.pth")
    
    # deterministically select DR trajectories
    rng = np.random.default_rng(args.seed)
    n_dr = int(n_trajectories * args.dr_fraction)
    
    # shuffle indices and take first n_dr for DR
    all_indices = np.arange(n_trajectories)
    rng.shuffle(all_indices)
    dr_indices = set(all_indices[:n_dr])
    
    print(f"Selected {n_dr} trajectories ({args.dr_fraction*100:.0f}%) for domain randomization")
    print(f"Remaining {n_trajectories - n_dr} trajectories will be canonical")
    
    # Metadata to save
    dr_metadata = {
        'dr_fraction': args.dr_fraction,
        'seed': args.seed,
        'dr_trajectory_indices': sorted(list(dr_indices)),
        'trajectory_configs': {},  # Will store per-trajectory DR configs
    }
    
    # Process each trajectory
    print(f"\nProcessing trajectories (background consistent per trajectory)...")
    n_with_distractor = 0
    n_without_distractor = 0
    for traj_idx in tqdm(range(n_trajectories), desc="Rendering"):
        seq_len = int(seq_lengths[traj_idx])
        traj_states = states[traj_idx, :seq_len].numpy()
        
        if traj_idx in dr_indices:
            # Apply domain randomization
            # Create a new RNG for this trajectory (deterministic based on traj_idx)
            traj_rng = np.random.default_rng(args.seed + traj_idx + 1000)
            
            rendered_images, trajectory_config = render_trajectory_with_dr(
                states=traj_states,
                seq_length=seq_len,
                rng=traj_rng,
                render_size=args.render_size,
                maze_spec=U_MAZE,
            )
            
            # Track distractor usage
            if trajectory_config.get('has_distractor', False):
                n_with_distractor += 1
            else:
                n_without_distractor += 1
            
            dr_metadata['trajectory_configs'][traj_idx] = trajectory_config
        else:
            # Canonical trajectory
            if args.re_render_canonical:
                rendered_images = render_trajectory_canonical(
                    states=traj_states,
                    seq_length=seq_len,
                    render_size=args.render_size,
                    maze_spec=U_MAZE,
                )
            else:
                # Copy original observations
                orig_obs_path = data_path / "obses" / f"episode_{traj_idx:03d}.pth"
                if orig_obs_path.exists():
                    rendered_images = torch.load(orig_obs_path).numpy()
                else:
                    # If original doesn't exist, re-render with canonical
                    print(f"  Warning: Original obs not found for traj {traj_idx}, re-rendering...")
                    rendered_images = render_trajectory_canonical(
                        states=traj_states,
                        seq_length=seq_len,
                        render_size=args.render_size,
                        maze_spec=U_MAZE,
                    )
        
        # Save rendered observations
        obs_tensor = torch.from_numpy(rendered_images)
        torch.save(obs_tensor, obses_path / f"episode_{traj_idx:03d}.pth")
    
    # Save metadata
    torch.save(dr_metadata, output_path / "dr_metadata.pth")
    
    print(f"\nDomain randomization dataset saved to: {output_path}")
    print(f"  - {n_dr} trajectories with DR (consistent background per trajectory)")
    if n_dr > 0:
        print(f"    * {n_with_distractor} with distractors ({n_with_distractor/n_dr*100:.1f}%)")
        print(f"    * {n_without_distractor} without distractors ({n_without_distractor/n_dr*100:.1f}%)")
    print(f"  - {n_trajectories - n_dr} canonical trajectories")
    print(f"  - Metadata saved to: {output_path / 'dr_metadata.pth'}")


if __name__ == '__main__':
    main()
