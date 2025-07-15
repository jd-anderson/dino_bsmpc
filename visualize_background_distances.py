import os
import sys
import warnings
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.pointmaze.maze_model import MazeEnv
from plan import ALL_MODEL_KEYS

# dictionary mapping model names to visualization titles
MODEL_TITLES = {
    # "bisim_100_mb_500_50_0_7": "Bisim 100; standard loss; memory buffer",
    "mod1_bisim_100_coef_1": "Bisim 100; standard loss; no memory buffer"
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

def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

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

        if result["encoder"].latent_ndim == 1:
            input_dim = result["encoder"].emb_dim
        else:
            decoder_scale = 16
            num_side_patches = train_cfg.img_size // decoder_scale
            num_patches = num_side_patches ** 2
            input_dim = num_patches * result["encoder"].emb_dim

        bisim_latent_dim = train_cfg.get('bisim_latent_dim', 64)
        bisim_hidden_dim = train_cfg.get('bisim_hidden_dim', 256)
        
        print(f"Creating bisimulation model with latent_dim={bisim_latent_dim}, hidden_dim={bisim_hidden_dim}")

        result["bisim_model"] = BisimModel(
            input_dim=input_dim,
            latent_dim=bisim_latent_dim,
            hidden_dim=bisim_hidden_dim,
            action_dim=train_cfg.action_emb_dim,
        )
        print(f"Created new bisimulation model with latent dim {bisim_latent_dim}")
    elif not train_cfg.get('has_bisim', False):
        result["bisim_model"] = None

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
        bisim_latent_dim=bisim_latent_dim,
        bisim_hidden_dim=bisim_hidden_dim,
        train_bisim=train_cfg.model.get('train_bisim', True),
    )
    model.to(device)
    return model

def load_model_from_path(model_path, device):
    """Load a model from the given path"""
    # load model config
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)
    
    # load checkpoint
    model_ckpt = Path(model_path) / "checkpoints" / "model_latest.pth"
    if not model_ckpt.exists():
        # try to find the highest numbered model
        ckpt_files = list((Path(model_path) / "checkpoints").glob("model_*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(f"No model checkpoints found in {model_path}")
        
        # find the highest numbered checkpoint
        numbered_ckpts = [f for f in ckpt_files if f.stem.startswith("model_") and f.stem[6:].isdigit()]
        if numbered_ckpts:
            model_ckpt = max(numbered_ckpts, key=lambda x: int(x.stem[6:]))
        else:
            model_ckpt = ckpt_files[0]
    
    print(f"Loading model from: {model_ckpt}")
    
    # load the model using our custom function
    num_action_repeat = model_cfg.num_action_repeat
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)
    
    return model, model_cfg

class FixedMazeEnv(MazeEnv):
    """MazeEnv with fixed dtype issues"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            site_id = self.model.site_name2id('target_site')
            rgba = self.model.site_rgba[site_id]
            rgba[-1] = 0.0
            self.model.site_rgba[site_id] = rgba
        except Exception:
            pass
        if hasattr(self, 'observation_space'):
            try:
                self.observation_space.dtype = np.float32
            except Exception:
                pass
        
        if len(self.goal_locations) == 1:
            self.set_target(np.array(self.goal_locations[0]).astype(np.float32))
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            self.set_target(np.array(self.reset_locations[0]).astype(np.float32))
    
    def _render_frame(self):
        if self.sim.render_contexts:
            cam = self.sim.render_contexts[0].cam
            cam.type      = 1
            cam.azimuth   = 90
            cam.elevation = -90
            cam.distance  = 4
        return self.sim.render(224, 224)
    
    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(np.float32)
            target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        target_location = np.array(target_location, dtype=np.float32)
        self._target = target_location

    def reset_model(self):
        idx = self.np_random.choice(len(self.empty_and_goal_locations))
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(np.float32)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(np.float32)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def set_marker(self):
        pass

def create_env_with_background(background_config):
    """Create environment with specific background"""
    env = FixedMazeEnv(
        return_value='state',
        background_builtin=background_config["builtin"],
        background_rgb1=background_config["rgb1"],
        background_rgb2=background_config["rgb2"],
        reset_target=False
    )
    return env

def generate_observations(state, backgrounds):
    """Generate observations for the same state with different backgrounds"""
    observations = {}
    
    for bg_name, bg_config in backgrounds.items():
        try:
            env = create_env_with_background(bg_config)
            env.seed(42)
            
            # set the environment to the specific state
            env.reset()
            env.set_state(state[:2], state[2:4])
            
            # get observation
            obs = env._get_obs()
            visual = env._render_frame()
            
            observations[bg_name] = {
                'visual': visual,
                'proprio': obs['proprio']
            }
            
            env.close()
            
        except Exception as e:
            print(f"    Error generating observation for background {bg_name}: {e}")
            continue
    
    return observations

def encode_observations(observations, model, device):
    """Encode observations using the model"""
    encodings = {}
    
    for bg_name, obs in observations.items():
        try:
            # prepare observation for model
            visual = torch.tensor(obs['visual']).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device) / 255.0
            proprio = torch.tensor(obs['proprio']).float().unsqueeze(0).unsqueeze(0).to(device)
            
            obs_dict = {
                'visual': visual,
                'proprio': proprio
            }
            
            # encode observation
            with torch.no_grad():
                model.eval()
                z_obs = model.encode_obs(obs_dict)
                
                # get DinoV2 embeddings
                dinov2_emb = z_obs['visual'].flatten(2)
                dinov2_emb = dinov2_emb.squeeze().cpu().numpy()
                
                # get bisimulation embeddings if available
                bisim_emb = None
                if model.has_bisim:
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

def compute_pca_and_distances(encodings, space_name):
    """Compute PCA and distances for a given encoding space"""
    # collect all embeddings
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
    
    fig, axes = plt.subplots(n_states, n_backgrounds, figsize=(4*n_backgrounds, 4*n_states))
    
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
                ax.set_title(f'{bg_config["title"]}\nState {state_idx+1}')
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{bg_config["title"]}\nState {state_idx+1}')
                ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_visualization(all_encodings, model_names, model_titles):
    """Create the complete visualization"""
    n_models = len(model_names)
    n_cols = n_models + 1  # +1 for DinoV2
    
    n_states = len(set(key.split('_')[1] for model_enc in all_encodings.values() for key in model_enc.keys()))
    if n_states == 0:
        n_states = 1

    fig, axes = plt.subplots(n_states, n_cols, figsize=(5*n_cols, 4*n_states))
    
    # handle different axes shapes
    if n_states == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_states == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    elif len(axes.shape) == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    
    # process each state
    for state_idx in range(n_states):
        # plot DinoV2 space
        ax = axes[state_idx, 0]
        
        # collect DinoV2 encodings for this state
        state_encodings = {}
        for bg_name in BACKGROUNDS.keys():
            # use first model's DinoV2 encoding (should be same across models)
            first_model = model_names[0]
            key = f"state_{state_idx}_{bg_name}"
            if first_model in all_encodings and key in all_encodings[first_model]:
                state_encodings[bg_name] = all_encodings[first_model][key]
        
        embeddings_2d, labels, distances = compute_pca_and_distances(state_encodings, 'dinov2')
        
        if embeddings_2d is not None and len(embeddings_2d) > 0:
            for i, (bg_name, emb_2d) in enumerate(zip(labels, embeddings_2d)):
                bg_config = BACKGROUNDS[bg_name]
                ax.scatter(emb_2d[0], emb_2d[1], c=colors[i % len(colors)], marker=markers[i % len(markers)], 
                          s=100, label=bg_config['title'], alpha=0.7)
            
            ax.set_title('DinoV2 Space')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.grid(True, alpha=0.3)
            
            # add legend below plot
            ax.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', fontsize=8, ncol=3, framealpha=0.9)
            
            # add distance text below legend
            if distances:
                dist_text = '\n'.join([f"{k}: {v:.3f}" for k, v in distances.items()])
                ax.text(0.5, -0.45, dist_text, transform=ax.transAxes,
                        horizontalalignment='center', verticalalignment='top', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No DinoV2 data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'DinoV2 Space')
        
        # plot bisimulation spaces
        for col_idx, model_name in enumerate(model_names):
            col = col_idx + 1
            ax = axes[state_idx, col]
            
            # get encodings for this model and state
            state_encodings = {}
            for bg_name in BACKGROUNDS.keys():
                key = f"state_{state_idx}_{bg_name}"
                if model_name in all_encodings and key in all_encodings[model_name]:
                    state_encodings[bg_name] = all_encodings[model_name][key]
            
            embeddings_2d, labels, distances = compute_pca_and_distances(state_encodings, 'bisim')
            
            if embeddings_2d is not None and len(embeddings_2d) > 0:
                for i, (bg_name, emb_2d) in enumerate(zip(labels, embeddings_2d)):
                    bg_config = BACKGROUNDS[bg_name]
                    ax.scatter(emb_2d[0], emb_2d[1], c=colors[i % len(colors)], marker=markers[i % len(markers)], 
                              s=100, label=bg_config['title'], alpha=0.7)
                
                title = model_titles.get(model_name, model_name)
                ax.set_title(f'{title}\nBisim Space')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.grid(True, alpha=0.3)
                
                ax.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', fontsize=8, ncol=3, framealpha=0.9)
                
                if distances:
                    dist_text = '\n'.join([f"{k}: {v:.3f}" for k, v in distances.items()])
                    ax.text(0.5, -0.45, dist_text, transform=ax.transAxes, 
                           horizontalalignment='center', verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            else:
                ax.text(0.5, 0.5, 'No bisim data', ha='center', va='center', transform=ax.transAxes)
                title = model_titles.get(model_name, model_name)
                ax.set_title(f'{title}\nBisim Space')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    return fig

def main():
    outputs_dir = "./outputs"
    if not os.path.exists(outputs_dir):
        print(f"Error: {outputs_dir} not found. Please run this script from the directory containing the outputs folder.")
        print("Or update the outputs_dir variable in the script to point to the correct location.")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    available_models = []
    for model_name in MODEL_TITLES.keys():
        model_path = os.path.join(outputs_dir, model_name)
        if os.path.exists(model_path):
            available_models.append(model_name)
        else:
            print(f"Warning: Model '{model_name}' not found in {model_path}")
    
    if not available_models:
        print("No specified models found in outputs folder.")
        print("Please check that the models listed in MODEL_TITLES exist in:", outputs_dir)
        print("Currently specified models:")
        for model_name in MODEL_TITLES.keys():
            print(f"  - {model_name}")
        return
    
    print(f"Found {len(available_models)} specified models:")
    for model_name in available_models:
        title = MODEL_TITLES.get(model_name, model_name)
        print(f"  - {model_name}: {title}")
    
    test_states = [
         np.array([2.5, 1.2, 0.0, 0.0]),
    ]
    
    # load models and generate encodings
    all_encodings = {}
    all_observations = {}  # store observations for visualization
    
    for model_name in available_models:
        print(f"\nProcessing model: {model_name}")
        model_path = os.path.join(outputs_dir, model_name)
        
        try:
            model, model_cfg = load_model_from_path(model_path, device)
            
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
    fig = create_visualization(all_encodings, available_models, MODEL_TITLES)
    
    # save and show
    filename = "distance_visualization.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved encoding visualization to {filename}")
    plt.show()
    
    print(f"\nVisualization complete!")
    print(f"Generated files:")
    print(f"  - {obs_filename}: Test observations with different backgrounds")
    print(f"  - {filename}: PCA visualization of encodings in DinoV2 and bisimulation spaces")

if __name__ == "__main__":
    main() 