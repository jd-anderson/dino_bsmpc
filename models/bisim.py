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
    
    def compute_correlation_distance(self, next_z_bisim, next_z_bisim2):
        """
        Compute correlation-based distance between next states using a more stable approach.
        
        Args:
            next_z_bisim: (batch_size, time_steps, bisim_dim) tensor
            next_z_bisim2: (batch_size, time_steps, bisim_dim) tensor
        
        Returns:
            distance: (batch_size,) tensor of correlation distances
        """
        batch_size, time_steps, bisim_dim = next_z_bisim.shape
        
        # Initialize distances tensor
        distances = torch.zeros(batch_size, device=next_z_bisim.device)
        
        # Process each batch element separately to compute covariance matrices
        for i in range(batch_size):
            # Get the bisimulation states for current batch element
            # Shape: (time_steps, bisim_dim)
            states_1 = next_z_bisim[i]  # (time_steps, bisim_dim)
            states_2 = next_z_bisim2[i]  # (time_steps, bisim_dim)
            
            # Center the data (subtract mean across time dimension)
            states_1_centered = states_1 - states_1.mean(dim=0, keepdim=True)
            states_2_centered = states_2 - states_2.mean(dim=0, keepdim=True)
            
            # Compute covariance matrices: (bisim_dim, bisim_dim)
            # cov = (1/n) * X^T * X where X is (time_steps, bisim_dim)
            cov_1 = torch.mm(states_1_centered.t(), states_1_centered) / time_steps
            cov_2 = torch.mm(states_2_centered.t(), states_2_centered) / time_steps
            
            # Add small epsilon to diagonal for numerical stability
            epsilon = 1e-6
            cov_1 = cov_1 + epsilon * torch.eye(bisim_dim, device=cov_1.device)
            cov_2 = cov_2 + epsilon * torch.eye(bisim_dim, device=cov_2.device)
            
            # Use a more stable approach: compute the Frobenius norm directly
            # This avoids the matrix square root which can be unstable during backprop
            diff_cov = cov_1 - cov_2
            distances[i] = torch.norm(diff_cov, p='fro')
            
            # Check for NaN or Inf in result
            if torch.isnan(distances[i]) or torch.isinf(distances[i]):
                print(f"WARNING: NaN or Inf in correlation distance computation for batch {i}")
                distances[i] = torch.tensor(1.0, device=distances.device)  # Safe fallback
        
        return distances

    def var_loss(self, z_bisim, var_target=0.5, epsilon=0):
        """
        Calculate variance loss (core)
        input: z_bisim: (b, t, bisim_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t)
        """

        # Compute variance
        var = z_bisim.var(dim=0) # (T,D)
        
        # Compute sqrt(var + epsilon)
        std = torch.sqrt(var + epsilon)
        
        # If NaN appears, fallback to using var directly
        nan_mask = torch.isnan(std)
        if nan_mask.any():
            # Replace NaN entries with var values
            std = var
            print(f"WARNING: NaN or Inf in Variance computation")
            
        # Compute max(0, var_target - std)
        loss = torch.relu(var_target - std)

        return loss.mean(dim=1) # reduce the dimension to (T)
    

    def calc_var_loss(self, z_bisim, next_z_bisim, var_target=1, epsilon=0):
        
        """
        Calculate variance loss with memory buffer
        input: z_bisim: (b, t, bisim_dim)
        next_z_bisim: (b, t, bisim_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t)
        """
    
        T_Plus_1_z_bisim = torch.cat([z_bisim, next_z_bisim], dim=0)

        # calculate the loss directly
        loss = self.var_loss(T_Plus_1_z_bisim, var_target, epsilon)

        return loss # dimension=(T+1), or memory sample+1

    
    def calc_bisim_loss(self, z_bisim, z_bisim2, reward, reward2, next_z_bisim, next_z_bisim2, discount=0.99, train_w_reward_loss=True):
        """
        Calculate bisimulation loss
        bisimulation metric: d(s1,s2) = |r(s1) - r(s2)| + γ · d(P(s1), P(s2)) + Variance Loss
        input: z_bisim, z_bisim2: (b, t, bisim_dim) or (b, bisim_dim)
               reward, reward2: (b, t, 1) or (b, 1)
               next_z_bisim, next_z_bisim2: (b, t, bisim_dim) or (b, bisim_dim)
        output: bisim_loss: (b, t) or (b,)
        """
        # print(f"BISIM LOSS CALC: State dim={z_bisim.shape[-1]}, calculating bisimilarity metric")
        # print(f"BISIM LOSS CALC: z_bisim={z_bisim.shape}, next_z_bisim={next_z_bisim.shape}")
        
        # Compute distance between current states
        z_dist = torch.sum(F.smooth_l1_loss(z_bisim, z_bisim2, reduction="none"), dim=-1)
        
        # Compute distance between rewards
        r_dist = torch.sum(F.smooth_l1_loss(reward, reward2, reduction="none"), dim=-1)
        
        
        # Compute correlation-based distance between next states
        transition_dist = self.compute_correlation_distance(next_z_bisim, next_z_bisim2)
        
        # Check for NaN or Inf in transition_dist
        if torch.isnan(transition_dist).any() or torch.isinf(transition_dist).any():
            print("WARNING: NaN or Inf values detected in transition_dist!")
            print(f"transition_dist shape: {transition_dist.shape}")
            print(f"transition_dist stats: mean={transition_dist.mean().item():.6f}, std={transition_dist.std().item():.6f}, min={transition_dist.min().item():.6f}, max={transition_dist.max().item():.6f}")
            # Replace with safe values
            transition_dist = torch.ones_like(transition_dist)
        
        # Expand transition_dist to match time dimension of r_dist
        # transition_dist: (b,) -> (b, t) by repeating for each time step
        transition_dist = transition_dist.unsqueeze(1).expand(-1, r_dist.shape[1])  # (b, t)

        var_loss = self.calc_var_loss(z_bisim, next_z_bisim, var_target=1, epsilon=0)
        
        
        # Target bisimilarity
        # if want reward_loss
        if train_w_reward_loss:
            target_bisimilarity = r_dist + discount * transition_dist
        else:
            target_bisimilarity = 0*r_dist + discount * transition_dist
        
        bisim_loss = (z_dist - target_bisimilarity).pow(2)

        print(bisim_loss)
        # Bisimulation loss
        bisim_loss = bisim_loss + var_loss
        print(bisim_loss)
        print()
        print(var_loss)
        
        # print(f"BISIM LOSS CALC: Final bisim loss shape={bisim_loss.shape}")
        
        return bisim_loss 