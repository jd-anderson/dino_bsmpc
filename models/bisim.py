import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np


def build_mlp(input_dim, hidden_dim, output_dim, num_hidden_layers):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ELU()]
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ELU()]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


class BisimModel(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim=256,
        num_hidden_layers=2,
        action_dim=10,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # Encoder from DinoV2 embeddings to bisimulation space
        self.encoder = build_mlp(input_dim, hidden_dim, latent_dim, num_hidden_layers)
        
        
        # Reward predictor
        self.reward = build_mlp(
            latent_dim + action_dim,  # latent state + action embedding
            hidden_dim,
            1,
            num_hidden_layers=1,
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def log_bisim(self, data):
        """Log bisimulation model details"""
        # Convert data to JSON-serializable format
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, np.integer):
                serializable_data[key] = int(value)
            elif isinstance(value, np.floating):
                serializable_data[key] = float(value)
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serializable_data[key] = value.detach().cpu().numpy().tolist()
            elif isinstance(value, (list, tuple)):
                # Handle lists/tuples that might contain numpy types
                serializable_data[key] = []
                for item in value:
                    if isinstance(item, np.integer):
                        serializable_data[key].append(int(item))
                    elif isinstance(item, np.floating):
                        serializable_data[key].append(float(item))
                    elif isinstance(item, np.ndarray):
                        serializable_data[key].append(item.tolist())
                    elif isinstance(item, torch.Tensor):
                        serializable_data[key].append(item.detach().cpu().numpy().tolist())
                    else:
                        serializable_data[key].append(item)
            elif isinstance(value, dict):
                # Handle nested dictionaries
                serializable_data[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.integer):
                        serializable_data[key][k] = int(v)
                    elif isinstance(v, np.floating):
                        serializable_data[key][k] = float(v)
                    elif isinstance(v, np.ndarray):
                        serializable_data[key][k] = v.tolist()
                    elif isinstance(v, torch.Tensor):
                        serializable_data[key][k] = v.detach().cpu().numpy().tolist()
                    else:
                        serializable_data[key][k] = v
            else:
                serializable_data[key] = value
        
        log_entry = {
            "timestamp": np.datetime64('now').astype(str),
            **serializable_data
        }
        with open("bisim_log.json", "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def encode(self, z_dino):
        """
        Maps DinoV2 embeddings to bisimulation embeddings
        input: z_dino: (b, t, p, d)
        output: z_bisim: (b, t, bisim_dim)
        """
        b, t, p, d = z_dino.shape
        # print(f"BISIM ENCODE: DinoV2 shape={z_dino.shape}, features={p*d}, bisim_dim={self.latent_dim}")
        z_dino = z_dino.reshape(b * t, p * d)
        z_bisim = self.encoder(z_dino)
        z_bisim = z_bisim.reshape(b, t, self.latent_dim)
        
        # Log encoding details
        self.log_bisim({
            "event": "bisim_encode",
            "input_shape": list(z_dino.shape),
            "input_reshaped_shape": list(z_dino.shape),
            "output_shape": list(z_bisim.shape),
            "input_stats": {
                "mean": float(z_dino.mean().item()),
                "std": float(z_dino.std().item()),
                "min": float(z_dino.min().item()),
                "max": float(z_dino.max().item()),
            },
            "output_stats": {
                "mean": float(z_bisim.mean().item()),
                "std": float(z_bisim.std().item()),
                "min": float(z_bisim.min().item()),
                "max": float(z_bisim.max().item()),
            },
        })
        
        return z_bisim
    
    def next(self, z_bisim, action_emb):
        """
        Predicts next bisimulation state
        input: z_bisim: (b, bisim_dim)
               action_emb: (b, action_emb_dim)
        output: next_z_bisim: (b, bisim_dim)
        """
        # print(f"BISIM NEXT: Bisim state shape={z_bisim.shape}, dim={z_bisim.shape[-1]}, action shape={action_emb.shape}")
        x = torch.cat([z_bisim, action_emb], dim=-1)
        result = self.dynamics(x)
        # print(f"BISIM NEXT: Output next state shape={result.shape}, dim={result.shape[-1]}")
        return result

    def predict_reward(self, z_bisim, action_emb):
        """
        Predicts reward from bisimulation state and action
        input: z_bisim: (b, bisim_dim)
               action_emb: (b, action_emb_dim)
        output: reward: (b, 1)
        """
        # print(f"DEBUG - BisimModel.predict_reward input dimensions: z_bisim {z_bisim.shape}, action_emb {action_emb.shape}")
        x = torch.cat([z_bisim, action_emb], dim=-1)
        # print(f"DEBUG - BisimModel.predict_reward concatenated input: {x.shape}")
        reward = self.reward(x)
        
        # Log reward prediction
        self.log_bisim({
            "event": "bisim_predict_reward",
            "z_bisim_shape": list(z_bisim.shape),
            "action_emb_shape": list(action_emb.shape),
            "concatenated_shape": list(x.shape),
            "reward_shape": list(reward.shape),
            "reward_stats": {
                "mean": float(reward.mean().item()),
                "std": float(reward.std().item()),
                "min": float(reward.min().item()),
                "max": float(reward.max().item()),
            },
        })
        
        return reward
    
    def calc_bisim_loss(self, z_bisim, z_bisim2, reward, reward2, next_z_bisim, next_z_bisim2, discount=0.99, train_w_reward_loss=True):
        """
        Calculate bisimulation loss
        bisimulation metric: d(s1,s2) = |r(s1) - r(s2)| + γ · d(P(s1), P(s2))
        """
        # print(f"BISIM LOSS CALC: State dim={z_bisim.shape[-1]}, calculating bisimilarity metric")
        # print(f"BISIM LOSS CALC: z_bisim={z_bisim.shape}, next_z_bisim={next_z_bisim.shape}")
        
        # Compute distance between current states
        z_dist = torch.sum(F.smooth_l1_loss(z_bisim, z_bisim2, reduction="none"), dim=-1)
        
        # Compute distance between rewards
        r_dist = torch.sum(F.smooth_l1_loss(reward, reward2, reduction="none"), dim=-1)
        
        # Compute distance between next states
        transition_dist = torch.norm(next_z_bisim - next_z_bisim2, dim=-1)
        
        # Target bisimilarity
        # if want reward_loss
        if train_w_reward_loss:
            target_bisimilarity = r_dist + discount * transition_dist
        else:
            target_bisimilarity = 0*r_dist + discount * transition_dist
        
        # Bisimulation loss
        bisim_loss = (z_dist - target_bisimilarity).pow(2)
        
        # Log bisimulation loss computation
        self.log_bisim({
            "event": "bisim_calc_loss",
            "z_bisim_shape": list(z_bisim.shape),
            "z_bisim2_shape": list(z_bisim2.shape),
            "reward_shape": list(reward.shape),
            "reward2_shape": list(reward2.shape),
            "next_z_bisim_shape": list(next_z_bisim.shape),
            "next_z_bisim2_shape": list(next_z_bisim2.shape),
            "z_dist_stats": {
                "mean": float(z_dist.mean().item()),
                "std": float(z_dist.std().item()),
            },
            "r_dist_stats": {
                "mean": float(r_dist.mean().item()),
                "std": float(r_dist.std().item()),
            },
            "transition_dist_stats": {
                "mean": float(transition_dist.mean().item()),
                "std": float(transition_dist.std().item()),
            },
            "target_bisimilarity_stats": {
                "mean": float(target_bisimilarity.mean().item()),
                "std": float(target_bisimilarity.std().item()),
            },
            "bisim_loss_stats": {
                "mean": float(bisim_loss.mean().item()),
                "std": float(bisim_loss.std().item()),
            },
            "discount": discount,
            "train_w_reward_loss": train_w_reward_loss,
        })
        
        # print(f"BISIM LOSS CALC: Final bisim loss shape={bisim_loss.shape}")
        
        return bisim_loss 