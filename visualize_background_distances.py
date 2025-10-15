import os
import sys
import warnings
from pathlib import Path
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def set_global_seed(seed=42):
    """Set all random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from env.pointmaze.point_maze_wrapper import PointMazeWrapper
from env.pointmaze.maze_model import U_MAZE
from plan import ALL_MODEL_KEYS

# dictionary mapping model names to visualization titles and optional epoch specification
# Format: "model_name": ("title", epoch_number) or "model_name": "title" (uses latest)
MODEL_TITLES = {
    # "bisim_100_mb_500_50_0_7": ("Bisim 100; memory buffer; epoch 25", 25),
    # "mod1_bisim_100_coef_1": ("Bisim 100; no memory buffer; epoch 50", 50),
    # "mod1_bisim_512_coef_1": "Bisim 512; no memory buffer",
    # "bisim_512_mb_1000_200": ("Bisim 512, epoch 25 mb 200", 25),
    # "mod1_bisim_1024_coef_1": "Bisim 1024; no memory buffer",
    # "mod1_bisim_5000_coef_1/21-59-56": "Bisim 5000; no memory buffer",
    "jepa_512": "Bisim 512",
    "jepa_512_vc_reg": "Bisim 512, VC Reg",
}

# background config
BACKGROUNDS = {
    "no_change": {
        "builtin": "checker",
        "rgb1": "0.2 0.3 0.4",
        "rgb2": "0.1 0.2 0.3",
        "title": "No Change"
    },
    "slight_change": {
        "builtin": "checker",
        "rgb1": "0.4 0.5 0.6",
        "rgb2": "0.3 0.4 0.5",
        "title": "Slight Change"
    },
    "gradient": {
        "builtin": "gradient",
        "rgb1": "0.2 0.3 0.4",
        "rgb2": "0.1 0.2 0.3",
        "title": "Gradient"
    }
}


def load_ckpt(snapshot_path, device):
    """Load model checkpoint"""
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model_with_bisim_params(model_ckpt, train_cfg, num_action_repeat, device):
    """Load model with properly configured bisimulation parameters"""
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Loaded checkpoint from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError("Decoder path not found in model checkpoint and is not provided in config")
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    # check if we need the bisimulation model
    if train_cfg.get('has_bisim', False) and "bisim_model" not in result:
        from models.bisim import BisimModel
        print('[DEBUG] bisim model not found in the checkpoint; creating a bisimulation model')
        if result["encoder"].latent_ndim == 1:  # if feature is 1D
            input_dim = result["encoder"].emb_dim
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = train_cfg.img_size // decoder_scale
            num_patches = num_side_patches ** 2
            input_dim = num_patches * result["encoder"].emb_dim

        result["bisim_model"] = BisimModel(
            input_dim=input_dim,
            latent_dim=train_cfg.get('bisim_latent_dim', 64),
            hidden_dim=train_cfg.get('bisim_hidden_dim', 256),
            action_dim=train_cfg.action_emb_dim,
        )
        print(f"Created new bisimulation model with latent dim {train_cfg.get('bisim_latent_dim', 64)}")
    elif not train_cfg.get('has_bisim', False):
        result["bisim_model"] = None

    # Extract actual bisimulation parameters from loaded model (if available)]
    if result.get("bisim_model") is not None:
        actual_bisim_latent_dim = result["bisim_model"].latent_dim
        actual_bisim_hidden_dim = result["bisim_model"].hidden_dim
        print(
            f"DEBUG: Using actual bisimulation parameters from checkpoint: latent_dim={actual_bisim_latent_dim}, hidden_dim={actual_bisim_hidden_dim}")
    else:
        actual_bisim_latent_dim = train_cfg.get('bisim_latent_dim', 64)
        actual_bisim_hidden_dim = train_cfg.get('bisim_hidden_dim', 256)
        print(
            f"DEBUG: Using config bisimulation parameters: latent_dim={actual_bisim_latent_dim}, hidden_dim={actual_bisim_hidden_dim}")

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        bisim_model=result.get("bisim_model", None),
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
        bisim_coef=train_cfg.get('bisim_coef', 1.0),
        bisim_latent_dim=actual_bisim_latent_dim,
        bisim_hidden_dim=actual_bisim_hidden_dim,
        train_bisim=train_cfg.model.get('train_bisim', True),
    )
    model.to(device)

    return model


def load_model_from_path(model_path, device, epoch=None):
    """Load a model from the given path, optionally specifying epoch"""
    # load model config
    try:
        with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
            model_cfg = OmegaConf.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load hydra.yaml: {e}")
        print(f"Skipping model due to config file error")
        raise e

    # load checkpoint
    if epoch is not None:
        model_ckpt = Path(model_path) / "checkpoints" / f"model_{epoch}.pth"
        if not model_ckpt.exists():
            print(f"Warning: Epoch {epoch} checkpoint not found, trying alternative names...")
            alternatives = [
                f"model_epoch_{epoch}.pth",
                f"checkpoint_{epoch}.pth",
                f"epoch_{epoch}.pth"
            ]
            for alt_name in alternatives:
                alt_path = Path(model_path) / "checkpoints" / alt_name
                if alt_path.exists():
                    model_ckpt = alt_path
                    break
            else:
                ckpt_dir = Path(model_path) / "checkpoints"
                available_ckpts = list(ckpt_dir.glob("*.pth"))
                print(f"Available checkpoints in {ckpt_dir}:")
                for ckpt in sorted(available_ckpts):
                    print(f"  - {ckpt.name}")
                raise FileNotFoundError(f"Epoch {epoch} checkpoint not found in {model_path}")
    else:
        model_ckpt = Path(model_path) / "checkpoints" / "model_latest.pth"
        if not model_ckpt.exists():
            ckpt_files = list((Path(model_path) / "checkpoints").glob("model_*.pth"))
            if not ckpt_files:
                raise FileNotFoundError(f"No model checkpoints found in {model_path}")

            numbered_ckpts = [f for f in ckpt_files if f.stem.startswith("model_") and f.stem[6:].isdigit()]
            if numbered_ckpts:
                model_ckpt = max(numbered_ckpts, key=lambda x: int(x.stem[6:]))
            else:
                model_ckpt = ckpt_files[0]

    print(f"Loading model from: {model_ckpt}")

    num_action_repeat = model_cfg.num_action_repeat
    model = load_model_with_bisim_params(model_ckpt, model_cfg, num_action_repeat, device=device)

    if epoch is not None:
        model_cfg.loaded_epoch = epoch
    else:
        try:
            if "model_latest" in str(model_ckpt):
                model_cfg.loaded_epoch = "latest"
            elif "model_" in str(model_ckpt):
                epoch_str = str(model_ckpt).split("model_")[1].split(".pth")[0]
                if epoch_str.isdigit():
                    model_cfg.loaded_epoch = int(epoch_str)
                else:
                    model_cfg.loaded_epoch = "unknown"
            else:
                model_cfg.loaded_epoch = "unknown"
        except:
            model_cfg.loaded_epoch = "unknown"

    return model, model_cfg


def create_env_with_background(background_config):
    """Create environment with specific background"""
    env = PointMazeWrapper(
        maze_spec=U_MAZE,
        return_value='state',
        background_builtin=background_config["builtin"],
        background_rgb1=background_config["rgb1"],
        background_rgb2=background_config["rgb2"],
        reset_target=False,
        with_target=True
    )
    return env


def generate_observations(state, backgrounds):
    """Generate observations for the same state with different backgrounds"""
    observations = {}

    print(f"    Generating observations for state: {state}")

    for bg_name, bg_config in backgrounds.items():
        try:
            print(f"      Processing background: {bg_name}")

            set_global_seed(42)

            env = create_env_with_background(bg_config)
            env.seed(42)

            # prepare environment for rendering
            env.prepare_for_render()

            # set the exact state
            env.set_state(state[:2], state[2:4])
            env.set_target(state[2:4])
            env.set_marker()

            # get observation
            obs = env._get_obs()
            visual = env._render_frame()

            obs_dict = {
                'visual': visual,
                'proprio': obs['proprio']
            }

            observations[bg_name] = obs_dict
            env.close()

        except Exception as e:
            print(f"    Error generating observation for background {bg_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return observations


def encode_observations(observations, model, device):
    """Encode observations using the model"""
    encodings = {}

    print(f"    Encoding {len(observations)} observations")

    for bg_name, obs in observations.items():
        try:
            print(f"      Encoding background: {bg_name}")
            set_global_seed(42)

            # prepare observation for model
            visual = torch.tensor(obs['visual']).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            proprio = torch.tensor(obs['proprio']).float().unsqueeze(0).unsqueeze(0).to(device)

            obs_dict = {
                'visual': visual,
                'proprio': proprio
            }

            with torch.no_grad():
                model.eval()
                bypass = getattr(model, 'bypass_dinov2', False)
                dinov2_emb = None
                bisim_emb = None

                if not bypass:
                    # get DinoV2 embeddings (directly from encoder forward pass)
                    b = visual.shape[0]
                    vis = model.encoder_transform(visual.view(b, visual.shape[2], visual.shape[3], visual.shape[4]))
                    dino_tokens = model.encoder.forward(vis)  # (b, p, d)
                    dinov2_emb = dino_tokens.flatten(1).squeeze().cpu().numpy()

                # get bisimulation embeddings
                if getattr(model, 'has_bisim', False):
                    if bypass:
                        z_bisim = model.encode_bisim(obs_dict)
                    else:
                        z_obs = model.encode_obs(obs_dict)
                        z_bisim = model.encode_bisim(z_obs)
                    bisim_emb = z_bisim.squeeze().cpu().numpy()

                encodings[bg_name] = {
                    'dinov2': dinov2_emb,
                    'bisim': bisim_emb
                }

        except Exception as e:
            print(f"    Error encoding observation for background {bg_name}: {e}")
            continue

    return encodings


def compute_pca_and_distances(encodings, space_name=None, compare_all=False):
    if compare_all:
        embeddings = []
        for state_key in encodings:
            embeddings.append((state_key, encodings[state_key]['bisim']))
        distances = {}
        for i in range(len(embeddings)):
            key1, emb1 = embeddings[i]
            for j in range(i + 1, len(embeddings)):
                key2, emb2 = embeddings[j]
                dist = np.linalg.norm(emb2 - emb1)
                distances[f'{key1}-{key2}'] = dist
        return distances

    embeddings = []
    labels = []

    for bg_name, enc in encodings.items():
        if enc[space_name] is not None:
            embeddings.append(enc[space_name])
            labels.append(bg_name)

    if len(embeddings) == 0:
        return None, None, None

    embeddings = np.array(embeddings)

    # apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # compute pairwise distances
    distances = {}
    for i, bg1 in enumerate(labels):
        for j, bg2 in enumerate(labels):
            if i < j:  # only compute upper triangle
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances[f"{bg1}-{bg2}"] = dist

    return embeddings_2d, labels, distances


def create_observation_visualization(all_observations, test_states):
    """Create visualization of actual observations"""
    n_states = len(test_states)
    n_backgrounds = len(BACKGROUNDS)

    fig, axes = plt.subplots(n_states, n_backgrounds, figsize=(4 * n_backgrounds, 4 * n_states))

    # handle different axes shapes
    if n_states == 1 and n_backgrounds == 1:
        axes = np.array([[axes]])
    elif n_states == 1:
        axes = axes.reshape(1, -1)
    elif n_backgrounds == 1:
        axes = axes.reshape(-1, 1)

    # plot observations
    for state_idx in range(n_states):
        for bg_idx, (bg_name, bg_config) in enumerate(BACKGROUNDS.items()):
            ax = axes[state_idx, bg_idx]

            if state_idx in all_observations and bg_name in all_observations[state_idx]:
                visual = all_observations[state_idx][bg_name]['visual']
                ax.imshow(visual)
                ax.set_title(f'{bg_config["title"]}\nState {state_idx + 1}')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{bg_config["title"]}\nState {state_idx + 1}')
                ax.axis('off')

    plt.tight_layout()
    return fig


def pretty_print_distances(distances):
    # extract unique state labels
    labels = set()
    for key in distances:
        a, b = key.split('-')
        labels.add(a)
        labels.add(b)
    labels = sorted(labels)

    # create symmetric distance DataFrame
    df = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for key, value in distances.items():
        a, b = key.split('-')
        df.at[a, b] = value
        df.at[b, a] = value  # symmetry

    # fill diagonal with zeros
    for label in labels:
        df.at[label, label] = 0.0

    print(df.fillna(''))


def create_visualization(all_encodings, model_names, model_titles):
    """Create one figure per model with two subplots: left=DINOv2, right=Bisim.
    Colors = states, Markers = backgrounds.
    """
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['red', 'blue', 'green', 'purple', 'orange', 'brown'])
    background_order = list(BACKGROUNDS.keys())
    default_markers = ['o', 's', '^', 'D', 'P', 'X']
    bg_marker_map = {bg: default_markers[i % len(default_markers)] for i, bg in enumerate(background_order)}

    figs = []
    for model_name in model_names:
        if model_name not in all_encodings:
            continue

        dino_pts, dino_states, dino_bgs = [], [], []
        bisim_pts, bisim_states, bisim_bgs = [], [], []
        model_enc = all_encodings[model_name]
        for key, enc in model_enc.items():
            # key format: 'state_{idx}_{bg_name}'
            try:
                _, state_str, bg_name = key.split('_', 2)
                state_id = int(state_str) if state_str.isdigit() else 0
            except Exception:
                state_id, bg_name = 0, 'unknown'
            if enc.get('dinov2') is not None:
                dino_pts.append(enc['dinov2'].reshape(-1))
                dino_states.append(state_id)
                dino_bgs.append(bg_name)
            if enc.get('bisim') is not None:
                bisim_pts.append(enc['bisim'].reshape(-1))
                bisim_states.append(state_id)
                bisim_bgs.append(bg_name)

        # Determine if we have any DINO points
        has_dino = len(dino_pts) > 0
        # Create figure: two subplots if DINO present, else only bisim subplot
        if has_dino:
            fig, (ax_dino, ax_bisim) = plt.subplots(1, 2, figsize=(12, 5))
            plt.subplots_adjust(bottom=0.22, wspace=0.25)
        else:
            fig, ax_bisim = plt.subplots(1, 1, figsize=(6, 5))
            plt.subplots_adjust(bottom=0.22)

        # DINO subplot
        if has_dino:
            X = np.vstack(dino_pts)
            X2 = PCA(n_components=2).fit_transform(X)
            unique_states = sorted(set(dino_states))
            state_to_color = {sid: color_cycle[i % len(color_cycle)] for i, sid in enumerate(unique_states)}
            present_bgs = sorted(set(dino_bgs))
            # jitter magnitude based on data spread
            if X2.shape[0] > 1:
                spread = max(X2[:, 0].std(), X2[:, 1].std())
            else:
                spread = 1.0
            jitter_r = 0.01 * spread if spread > 0 else 1e-3
            # deterministic offsets per (state, background)
            def sb_offset(sid, bg):
                idx = background_order.index(bg) if bg in background_order else 0
                code = (sid * 131 + idx * 17) % 360
                angle = 2 * np.pi * (code / 360.0)
                return np.array([np.cos(angle), np.sin(angle)]) * jitter_r
            for pt, sid, bg in zip(X2, dino_states, dino_bgs):
                off = sb_offset(sid, bg)
                ax_dino.scatter(pt[0] + off[0], pt[1] + off[1],
                                c=state_to_color[sid], marker=bg_marker_map.get(bg, 'o'),
                                s=70, alpha=0.7, edgecolors='k', linewidths=0.3)
            ax_dino.set_title('DINOv2 Space (all states)')
            ax_dino.grid(True, alpha=0.3)
        elif has_dino is False:
            # No DINO axis when bypassing or no data; nothing to draw here
            pass

        # Bisim subplot
        if bisim_pts:
            X = np.vstack(bisim_pts)
            X2 = PCA(n_components=2).fit_transform(X)
            unique_states = sorted(set(bisim_states))
            state_to_color = {sid: color_cycle[i % len(color_cycle)] for i, sid in enumerate(unique_states)}
            present_bgs = sorted(set(bisim_bgs))
            if X2.shape[0] > 1:
                spread = max(X2[:, 0].std(), X2[:, 1].std())
            else:
                spread = 1.0
            jitter_r = 0.01 * spread if spread > 0 else 1e-3
            def sb_offset(sid, bg):
                idx = background_order.index(bg) if bg in background_order else 0
                code = (sid * 131 + idx * 17) % 360
                angle = 2 * np.pi * (code / 360.0)
                return np.array([np.cos(angle), np.sin(angle)]) * jitter_r
            for pt, sid, bg in zip(X2, bisim_states, bisim_bgs):
                off = sb_offset(sid, bg)
                ax_bisim.scatter(pt[0] + off[0], pt[1] + off[1],
                                 c=state_to_color[sid], marker=bg_marker_map.get(bg, 'o'),
                                 s=70, alpha=0.7, edgecolors='k', linewidths=0.3)
            title = model_titles.get(model_name, model_name)
            ax_bisim.set_title(f'Bisimulation Space (all states)\n{title}')
            ax_bisim.grid(True, alpha=0.3)
        else:
            ax_bisim.text(0.5, 0.5, 'No bisim data', ha='center', va='center', transform=ax_bisim.transAxes)

        if has_dino:
            unique_states_leg = sorted(set(dino_states))
            unique_bgs_leg = background_order  # fixed order legends
            state_to_color_leg = {sid: color_cycle[i % len(color_cycle)] for i, sid in enumerate(unique_states_leg)}
        elif bisim_pts:
            unique_states_leg = sorted(set(bisim_states))
            unique_bgs_leg = background_order
            state_to_color_leg = {sid: color_cycle[i % len(color_cycle)] for i, sid in enumerate(unique_states_leg)}
        else:
            unique_states_leg = []
            unique_bgs_leg = []
            state_to_color_leg = {}

        state_handles = [plt.Line2D([], [], color=state_to_color_leg[sid], marker='o', linestyle='', label=f'State {sid}') for sid in unique_states_leg]
        bg_to_marker_leg = {bg: bg_marker_map.get(bg, 'o') for bg in unique_bgs_leg}
        # legend label: use BACKGROUNDS title if available, else bg key
        bg_handles = [plt.Line2D([], [], color='gray', marker=bg_to_marker_leg[bg], linestyle='',
                                  label=BACKGROUNDS.get(bg, {'title': bg}).get('title', bg)) for bg in unique_bgs_leg]

        if state_handles:
            anchor_x = 0.3 if has_dino else 0.25
            fig.legend(handles=state_handles, title='States', loc='lower center', bbox_to_anchor=(anchor_x, -0.02), ncol=min(len(state_handles), 6))
        if bg_handles:
            anchor_x = 0.75 if has_dino else 0.75
            fig.legend(handles=bg_handles, title='Backgrounds', loc='lower center', bbox_to_anchor=(anchor_x, -0.02), ncol=min(len(bg_handles), 6))

        figs.append(fig)

    return figs


def list_available_checkpoints(model_path):
    """List all available checkpoints for a model"""
    ckpt_dir = Path(model_path) / "checkpoints"
    if not ckpt_dir.exists():
        return []

    checkpoints = []
    for ckpt_file in ckpt_dir.glob("*.pth"):
        name = ckpt_file.stem
        epoch = None
        if name.startswith("model_") and name[6:].isdigit():
            epoch = int(name[6:])
        elif name == "model_latest":
            epoch = "latest"
        checkpoints.append((ckpt_file.name, epoch))

    return sorted(checkpoints, key=lambda x: x[1] if isinstance(x[1], int) else float('inf'))


def main():
    set_global_seed(42)
    print("Random seeds set to 42")

    outputs_dir = "./outputs"
    if not os.path.exists(outputs_dir):
        print(
            f"Error: {outputs_dir} not found. Please run this script from the directory containing the outputs folder.")
        print("Or update the outputs_dir variable in the script to point to the correct location.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # parse MODEL_TITLES to handle both string and tuple formats
    model_configs = {}
    for model_name, config in MODEL_TITLES.items():
        if isinstance(config, tuple):
            title, epoch = config
            model_configs[model_name] = {'title': title, 'epoch': epoch}
        else:
            model_configs[model_name] = {'title': config, 'epoch': None}

    available_models = []
    for model_name in model_configs.keys():
        model_path = os.path.join(outputs_dir, model_name)
        if os.path.exists(model_path):
            available_models.append(model_name)
        else:
            print(f"Warning: Model '{model_name}' not found in {model_path}")

    if not available_models:
        print("No specified models found in outputs folder.")
        print("Please check that the models listed in MODEL_TITLES exist in:", outputs_dir)
        print("Currently specified models:")
        for model_name in model_configs.keys():
            print(f"  - {model_name}")
        return

    print(f"Found {len(available_models)} specified models:")
    for model_name in available_models:
        config = model_configs[model_name]
        epoch_info = f" (epoch {config['epoch']})" if config['epoch'] is not None else " (latest)"
        print(f"  - {model_name}: {config['title']}{epoch_info}")

        model_path = os.path.join(outputs_dir, model_name)
        checkpoints = list_available_checkpoints(model_path)
        if checkpoints:
            print(
                f"    Available checkpoints: {[f'epoch {epoch}' if isinstance(epoch, int) else epoch for _, epoch in checkpoints]}")
        else:
            print(f"    No checkpoints found in {model_path}/checkpoints/")

    test_states = [
        np.array([2.5, 3.0, 3.0, 2.0]),
        np.array([3.2, 2, 1.0, 1.0]),
        np.array([2.5, 1.2, 1.0, 2.5]),
    ]

    # load models and generate encodings
    all_encodings = {}
    all_observations = {}  # store observations for visualization

    for model_name in available_models:
        config = model_configs[model_name]
        epoch_info = f" (epoch {config['epoch']})" if config['epoch'] is not None else " (latest)"
        print(f"\nProcessing model: {model_name}{epoch_info}")
        model_path = os.path.join(outputs_dir, model_name)

        try:
            model, model_cfg = load_model_from_path(model_path, device, epoch=config['epoch'])
            print(f"  Successfully loaded model from epoch {model_cfg.loaded_epoch}")

            # generate encodings for each test state
            model_encodings = {}

            for state_idx, state in enumerate(test_states):
                print(f"  Generating observations for state {state_idx + 1}")
                observations = generate_observations(state, BACKGROUNDS)

                # store observations for visualization
                if state_idx not in all_observations:
                    all_observations[state_idx] = observations

                print(f"  Encoding observations for state {state_idx + 1}")
                encodings = encode_observations(observations, model, device)

                # store encodings with state information
                for bg_name, enc in encodings.items():
                    key = f"state_{state_idx}_{bg_name}"
                    model_encodings[key] = enc

            all_encodings[model_name] = model_encodings

        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue

    if not all_encodings:
        print("No models were successfully loaded.")
        return

    # create observation visualization first
    print("\nCreating observation visualization")
    obs_fig = create_observation_visualization(all_observations, test_states)
    obs_filename = "test_observations.png"
    obs_fig.savefig(obs_filename, dpi=300, bbox_inches='tight')
    print(f"Saved observation visualization to {obs_filename}")
    plt.show()

    # create visualization
    print("\nCreating encoding visualization")
    model_titles = {name: config['title'] for name, config in model_configs.items()}
    figs = create_visualization(all_encodings, available_models, model_titles)

    # save and show
    for fig, model_name in zip(figs, available_models):
        safe_name = model_name.replace(os.sep, '_')
        filename = f"distance_visualization_{safe_name}.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved encoding visualization to {filename}")
        plt.show()

    print(f"\nVisualization complete!")
    print(f"Generated files:")
    print(f"  - {obs_filename}: Test observations with different backgrounds")
    print(f"  - {filename}: PCA visualization of encodings in DinoV2 and bisimulation spaces")

    for model_name in all_encodings.keys():
        actual_model_name = model_name
        if model_name == 'bisim_100_mb_500_50_0_7':
            actual_model_name = 'bisim_100_mb_1000_100'
        print(f'\ncomparing distances for the model {actual_model_name}\n')
        distances = compute_pca_and_distances(all_encodings[model_name], compare_all=True)
        pretty_print_distances(distances)
        print('\n')


if __name__ == "__main__":
    main()
