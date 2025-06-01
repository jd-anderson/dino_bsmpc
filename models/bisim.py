import torch
import torch.nn as nn
import torch.nn.functional as F


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
        
        # mod 1 function on BSMPC
        '''
        # Dynamics model to predict next state in bisimulation space
        self.dynamics = build_mlp(
            latent_dim + action_dim,  # latent state + action embedding
            hidden_dim,
            latent_dim,
            num_hidden_layers=1,
        )
        '''
        
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
        return self.reward(x)
    
    def calc_bisim_loss(self, z_bisim, z_bisim2, reward, reward2, next_z_bisim, next_z_bisim2, discount=0.99):
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
        target_bisimilarity = r_dist + discount * transition_dist
        
        # Bisimulation loss
        bisim_loss = (z_dist - target_bisimilarity).pow(2)
        
        # print(f"BISIM LOSS CALC: Final bisim loss shape={bisim_loss.shape}")
        
        return bisim_loss 