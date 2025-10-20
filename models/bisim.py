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
        bypass_dinov2=False,
        img_size=224,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.bypass_dinov2 = bypass_dinov2
        self.img_size = img_size
        
        if bypass_dinov2:
            # Direct encoding: raw observations (3 * img_size * img_size) -> bisim
            actual_input_dim = 3 * img_size * img_size
        else:
            # DinoV2 encoding: DinoV2 embeddings (patches * embed_dim) -> bisim
            actual_input_dim = input_dim
            
        self.encoder = build_mlp(actual_input_dim, hidden_dim, latent_dim, num_hidden_layers)
        
        # Reward predictor
        self.reward = build_mlp(
            latent_dim + action_dim,  # latent state + action embedding
            hidden_dim,
            1,
            num_hidden_layers=1,
        )
        
        # Initialize weights
        self._initialize_weights()

        self.PCAMatrix = []
        self.PCA_Calced = False
    
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
    
    def encode(self, input_data):
        """
        Maps input to bisimulation embeddings
        input: 
        - If bypass_dinov2=False: z_dino: (b, t, p, d) - DinoV2 embeddings
        - If bypass_dinov2=True: obs: (b, t, 3, img_size, img_size) - Raw observations
        output: z_bisim: (b, t, bisim_dim)
        """
        if self.bypass_dinov2:
            b, t, c, h, w = input_data.shape
            input_flat = input_data.reshape(b * t, c * h * w)
        else:
            b, t, p, d = input_data.shape
            input_flat = input_data.reshape(b * t, p * d)
        
        z_bisim = self.encoder(input_flat)
        z_bisim = z_bisim.reshape(b, t, self.latent_dim)
        
        # Log encoding details
        self.log_bisim({
            "event": "bisim_encode_direct" if self.bypass_dinov2 else "bisim_encode_dinov2",
            "input_shape": list(input_data.shape),
            "input_flattened_shape": list(input_flat.shape),
            "output_shape": list(z_bisim.shape),
            "bypass_dinov2": self.bypass_dinov2,
            "input_stats": {
                "mean": float(input_data.mean().item()),
                "std": float(input_data.std().item()),
                "min": float(input_data.min().item()),
                "max": float(input_data.max().item()),
            },
            "output_stats": {
                "mean": float(z_bisim.mean().item()),
                "std": float(z_bisim.std().item()),
                "min": float(z_bisim.min().item()),
                "max": float(z_bisim.max().item()),
            },
        })
        
        return z_bisim
    

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

    def compute_transition_distance(self, next_z_bisim, next_z_bisim2):
        """
        Compute distance between next state distributions for bisimulation metric.
        Uses L2 distance between predicted next states averaged over time.

        Args:
            next_z_bisim: (batch_size, time_steps, bisim_dim) tensor
            next_z_bisim2: (batch_size, time_steps, bisim_dim) tensor

        Returns:
            distance: (batch_size,) tensor of transition distances
        """
        # Compute L2 distance between next state predictions
        # Average over time dimension to get distance per batch element
        diff = next_z_bisim - next_z_bisim2  # (batch_size, time_steps, bisim_dim)
        squared_diff = diff.pow(2).sum(dim=-1)  # (batch_size, time_steps)
        distances = squared_diff.mean(dim=-1)  # (batch_size,) - average over time
        
        # Take square root to get L2 distance
        distances = torch.sqrt(distances + 1e-8)  # Add epsilon for numerical stability
        
        return distances

    def compute_covariance_regularization(self, z_bisim, next_z_bisim):
        """
        Compute covariance regularization for bisimulation embeddings.
        Encourages diverse and structured representations.
        
        Args:
            z_bisim: (batch_size, time_steps, bisim_dim) current states
            next_z_bisim: (batch_size, time_steps, bisim_dim) next states
            
        Returns:
            cov_reg: (batch_size,) tensor of covariance regularization loss
        """
        batch_size, time_steps, bisim_dim = z_bisim.shape
        
        # Combine current and next states for covariance computation
        combined_states = torch.cat([z_bisim, next_z_bisim], dim=0)  # (2*batch_size, time_steps, bisim_dim)
        
        # Reshape to treat each time step as a separate sample
        # (2*batch_size * time_steps, bisim_dim)
        states_flat = combined_states.reshape(-1, bisim_dim)
        
        # Center the data
        states_centered = states_flat - states_flat.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix: (bisim_dim, bisim_dim)
        n_samples = states_centered.shape[0]
        cov_matrix = torch.mm(states_centered.t(), states_centered) / (n_samples - 1)
        
        # Add small epsilon for numerical stability
        epsilon = 1e-6
        cov_matrix = cov_matrix + epsilon * torch.eye(bisim_dim, device=cov_matrix.device)
        
        # Compute covariance regularization loss
        # Encourage off-diagonal elements to be small (decorrelation)
        off_diag_mask = ~torch.eye(bisim_dim, dtype=torch.bool, device=cov_matrix.device)
        off_diag_loss = cov_matrix[off_diag_mask].pow(2).mean()
        
        # Encourage diagonal elements to be close to target variance
        diag_elements = torch.diag(cov_matrix)
        var_target = 1.0
        diag_loss = (diag_elements - var_target).pow(2).mean()
        
        # Total covariance regularization
        cov_reg = off_diag_loss + diag_loss
        
        # Return per-batch element (broadcast to match batch size)
        return cov_reg.expand(batch_size)

    def var_loss(self, z_bisim, var_target=0.1, epsilon=0):
        """
        Calculate variance loss (core)
        input: z_bisim: (b, t, bisim_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t)
        """

        # Compute variance
        var = z_bisim.var(dim=0)  # (T,D)

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

        return loss.mean(dim=1)  # reduce the dimension to (T)

    def cal_pca(self, z_bisim):
        B, T, D = z_bisim.shape
        Z = z_bisim.reshape(B * T, D)

        # Center data
        Z_centered = Z - Z.mean(dim=0, keepdim=True)

        # PCA via SVD
        U, S, Vt = torch.linalg.svd(Z_centered, full_matrices=False) # check  Vt or V
        V = Vt.T  # (D, D)

        self.PCAMatrix = V.detach()
        self.PCA_Calced = True

    def pca_var_loss(self, z_bisim, target_first=0.01, target_rest=2.0, num_pcs=10):
        """
        PCA variance loss:
        - 1st PC variance -> target_first
        - Next (num_pcs-1) PCs variance -> target_rest
        - Remaining PCs are unconstrained

        Args:
            z_bisim: Tensor (B, T, D)
            target_first: variance target for the first PC
            target_rest: variance target for PCs 2..num_pcs
            num_pcs: number of PCs to regularize
        Returns:
            scalar loss
        """
        B, T, D = z_bisim.shape
        Z = z_bisim.reshape(B * T, D)

        # Center data
        Z_centered = Z - Z.mean(dim=0, keepdim=True)

        if not self.PCA_Calced:
            self.cal_pca(z_bisim)
        V = self.PCAMatrix
        num_pcs = min(num_pcs, V.shape[1])
        V_10 = V[:, :num_pcs]  # (D, num_pcs)

        # Project to PCA coords
        Z_proj = Z_centered @ V_10  # (num_pcs)
        var_V10 = torch.zeros(num_pcs, device=Z_proj.device)

        # print(Z_proj)
        for i in range(num_pcs):
            var_V10[i] = Z_proj[:, i].var(unbiased=True)


        # Build targets
        targets = torch.full_like(var_V10, target_rest)
        if num_pcs > 0:
            targets[0] = target_first

        # Loss = error between pc_var and targets
        loss = (torch.abs(var_V10 - targets)).mean()

        return loss

    def calc_var_loss(self, z_bisim, next_z_bisim, var_target=0.1, epsilon=0):

        """
        Calculate variance loss with memory buffer
        input: z_bisim: (b, t, bisim_dim)
        next_z_bisim: (b, t, bisim_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t)
        """

        T_Plus_1_z_bisim = torch.cat([z_bisim, next_z_bisim], dim=0)

        # calculate the loss
        loss = self.var_loss(T_Plus_1_z_bisim, var_target, epsilon)
        #loss = self.pca_var_loss(T_Plus_1_z_bisim, target_first, var_target, num_pcs)

        return loss  # dimension=(T+1), or memory sample+1
    
    def calc_PCAVar_loss(self, z_bisim, next_z_bisim, target_first= 0.01, var_target=0.1, num_pcs=10):

        """
        Calculate variance loss with memory buffer
        input: z_bisim: (b, t, bisim_dim)
        next_z_bisim: (b, t, bisim_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t)
        """

        T_Plus_1_z_bisim = torch.cat([z_bisim, next_z_bisim], dim=0)

        # calculate the loss
        # loss = self.var_loss(T_Plus_1_z_bisim, var_target, epsilon)
        loss = self.pca_var_loss(T_Plus_1_z_bisim, target_first, var_target, num_pcs)

        return loss  # dimension=(T+1), or memory sample+1

    def calc_bisim_loss(self, z_bisim, z_bisim2, reward, reward2, next_z_bisim, next_z_bisim2, epoch, discount=0.99,
                        train_w_reward_loss=True, var_loss_coef: float = 1.0, PCA1_loss_target: float = 0.01, VC_target: float = 1.0,
                        num_pcs: int = 10, PCAloss_epoch: int = 50):
        """
        Calculate bisimulation loss
        bisimulation metric: d(s1,s2) = |r(s1) - r(s2)| + γ · d(P(s1), P(s2)) + Variance Loss + Covariance Regularization
        input: z_bisim, z_bisim2: (b, t, bisim_dim) or (b, bisim_dim)
               reward, reward2: (b, t, 1) or (b, 1)
               next_z_bisim, next_z_bisim2: (b, t, bisim_dim) or (b, bisim_dim)
        output: bisim_loss: (b, t) or (b,)
        bisimulation metric: d(s1,s2) = |r(s1) - r(s2)| + γ · d(P(s1), P(s2)) + Variance Loss + Covariance Regularization
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

        # Compute transition distance between next states
        transition_dist = self.compute_transition_distance(next_z_bisim, next_z_bisim2)

        # Check for NaN or Inf in transition_dist
        if torch.isnan(transition_dist).any() or torch.isinf(transition_dist).any():
            print("WARNING: NaN or Inf values detected in transition_dist!")
            print(f"transition_dist shape: {transition_dist.shape}")
            print(
                f"transition_dist stats: mean={transition_dist.mean().item():.6f}, std={transition_dist.std().item():.6f}, min={transition_dist.min().item():.6f}, max={transition_dist.max().item():.6f}")
            # Replace with safe values
            transition_dist = torch.ones_like(transition_dist)

        # Expand transition_dist to match time dimension of r_dist
        # transition_dist: (b,) -> (b, t) by repeating for each time step
        transition_dist = transition_dist.unsqueeze(1).expand(-1, r_dist.shape[1])  # (b, t)

        # Compute variance loss
        if epoch <= PCAloss_epoch:
            var_loss = self.calc_var_loss(z_bisim, next_z_bisim, VC_target, epsilon=0.1)
        else:
            var_loss = self.calc_PCAVar_loss(z_bisim, next_z_bisim, PCA1_loss_target, VC_target, num_pcs)
        
        # Compute covariance regularization
        cov_reg = self.compute_covariance_regularization(z_bisim, next_z_bisim)
        # Expand to match time dimension
        cov_reg = cov_reg.unsqueeze(1).expand(-1, r_dist.shape[1])  # (b, t)

        var_loss = var_loss * var_loss_coef
        cov_reg = cov_reg * var_loss_coef

        # Target bisimilarity
        # if want reward_loss
        if train_w_reward_loss:
            target_bisimilarity = r_dist + discount * transition_dist
        else:
            target_bisimilarity = 0 * r_dist + discount * transition_dist

        bisim_loss = (z_dist - target_bisimilarity).pow(2)

        # Bisimulation loss with variance and covariance regularization
        bisim_loss = bisim_loss + var_loss + cov_reg

        # print(f"BISIM LOSS CALC: Final bisim loss shape={bisim_loss.shape}")

        return bisim_loss, z_dist, r_dist, discount * transition_dist, var_loss, cov_reg
