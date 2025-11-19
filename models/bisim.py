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


def build_patch_encoder(input_dim, hidden_dim, output_dim, num_hidden_layers=1):
    """Build a small MLP for processing patches or blocks"""
    layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


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
        num_patches=196,  # number of output patches
        patch_emb_dim=384,  # DINOv2 patch embedding dimension
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.bypass_dinov2 = bypass_dinov2
        self.img_size = img_size
        self.num_patches = num_patches
        self.patch_dim = latent_dim
        self.patch_emb_dim = patch_emb_dim

        # check if we should use patch processing
        self.use_patch_processing = not bypass_dinov2

        if bypass_dinov2:
            # direct encoding: raw observations -> bisim
            actual_input_dim = 3 * img_size * img_size
            self.encoder = build_mlp(actual_input_dim, hidden_dim, latent_dim, num_hidden_layers)
        elif self.use_patch_processing:
            # 384 -> 128 -> ResBlock(128) -> 64
            middle_dim = 2 * self.patch_dim
            self.encoder = nn.Sequential(
                nn.Linear(patch_emb_dim, middle_dim),
                ResBlock(middle_dim),
                nn.Linear(middle_dim, self.patch_dim),
            )

            # spatial positional embedding for output patches
            self.spatial_pos_emb = nn.Parameter(torch.randn(num_patches, self.patch_dim))

            # layer norm after projection
            self.proj_norm = nn.LayerNorm(self.patch_dim)
        else:
            # legacy: flatten all patches
            actual_input_dim = input_dim
            self.encoder = build_mlp(actual_input_dim, hidden_dim, latent_dim, num_hidden_layers)

        reward_hidden_dim = (self.patch_dim + self.action_dim) * 2
        self.reward = build_mlp(self.patch_dim + self.action_dim, reward_hidden_dim, 1, num_hidden_layers=1)

        # aggregator: per-patch score -> softmax weights
        self.reward_aggregator = nn.Linear(self.patch_dim, 1)

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
        output: z_bisim: (b, t, num_patches, patch_dim)
        """
        if self.bypass_dinov2:
            b, t, c, h, w = input_data.shape
            input_flat = input_data.reshape(b * t, c * h * w)
            z_bisim = self.encoder(input_flat)  # (b*t, latent_dim)
            z_bisim = z_bisim.reshape(b, t, self.latent_dim)
            # (b, t, latent_dim) -> (b, t, num_patches, patch_dim)
            z_bisim = z_bisim.unsqueeze(2).expand(b, t, self.num_patches, self.patch_dim)
            event = "bisim_encode_direct"

        elif self.use_patch_processing:
            # apply per-token encoding: 384 -> patch_dim
            z_bisim = self.encoder(input_data)  # (b, t, 196, patch_dim)

            # add spatial positional embeddings
            z_bisim = z_bisim + self.spatial_pos_emb.unsqueeze(0).unsqueeze(0)  # broadcast over batch and time

            # apply layer norm
            z_bisim = self.proj_norm(z_bisim)

            event = "bisim_encode_dinov2_patches"

        else:
            # legacy: flatten all patches
            b, t, p, d = input_data.shape
            input_flat = input_data.reshape(b * t, p * d)
            z_bisim = self.encoder(input_flat)
            z_bisim = z_bisim.reshape(b, t, self.latent_dim)
            z_bisim = z_bisim.reshape(b, t, self.num_patches, self.patch_dim)
            event = "bisim_encode_dinov2_flat_legacy"

        # Log encoding details
        self.log_bisim({
            "event": event,
            "input_shape": list(input_data.shape),
            "output_shape": list(z_bisim.shape),
            "bypass_dinov2": self.bypass_dinov2,
            "use_patch_processing": self.use_patch_processing,
            "num_patches": self.num_patches,
            "patch_dim": self.patch_dim,
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
        z_bisim: (b, p, d) or (b, t, p, d) where d == self.patch_dim
        action_emb: (b, a) or (b, t, a)
        returns: (b, 1) or (b, t, 1)
        """
        assert z_bisim.shape[-1] == self.patch_dim, \
            f"z_bisim last dim {z_bisim.shape[-1]} must equal patch_dim {self.patch_dim}"

        if z_bisim.dim() == 4:
            b, t, p, d = z_bisim.shape
            z2 = z_bisim.reshape(b * t, p, d)  # (bt, p, d)
            scores = self.reward_aggregator(z2)  # (bt, p, 1)
            weights = torch.softmax(scores, dim=1)  # (bt, p, 1)
            z_agg = (z2 * weights).sum(dim=1)  # (bt, d)
            # action handling
            if action_emb is None:
                a = torch.zeros(b, t, self.action_dim, device=z_bisim.device, dtype=z_bisim.dtype)
            else:
                a = action_emb
            if a.dim() == 2:  # (b, a) -> (b, t, a)
                a = a.unsqueeze(1).expand(b, t, -1)
            a = a.reshape(b * t, -1)  # (bt, a)
            x = torch.cat([z_agg, a], dim=-1)  # (bt, d+a)
            out = self.reward(x).reshape(b, t, 1)  # (b, t, 1)
            return out

        elif z_bisim.dim() == 3:
            b, p, d = z_bisim.shape
            scores = self.reward_aggregator(z_bisim)  # (b, p, 1)
            weights = torch.softmax(scores, dim=1)  # (b, p, 1)
            z_agg = (z_bisim * weights).sum(dim=1)  # (b, d)
            # action handling
            if action_emb is None:
                a = torch.zeros(b, self.action_dim, device=z_bisim.device, dtype=z_bisim.dtype)
            else:
                a = action_emb
            if a.dim() == 3:  # (b, t, a) -> (b, a) not allowed here
                raise ValueError("predict_reward got (b,p,d) states but (b,t,a) actions.")
            x = torch.cat([z_agg, a], dim=-1)  # (b, d+a)
            out = self.reward(x)  # (b, 1)
            return out

        else:
            raise ValueError(f"z_bisim must be (b,p,d) or (b,t,p,d), got {z_bisim.shape}")

    def compute_transition_distance(self, next_z_bisim, next_z_bisim2):
        """
        Per-sequence transition distance.
        next_z_bisim:  (b, t, p, d)
        next_z_bisim2: (b, t, p, d)
        Returns: (b,) transition distance
        """
        b, t, p, d = next_z_bisim.shape
        z1_flat = next_z_bisim.reshape(b, t, p * d)  # (b, t, D)
        z2_flat = next_z_bisim2.reshape(b, t, p * d)  # (b, t, D)

        diff = z1_flat - z2_flat  # (b, t, D)
        squared_diff = diff.pow(2).sum(dim=-1)  # (b, t)
        distances = squared_diff.mean(dim=-1)  # (b,)  # average over time
        distances = torch.sqrt(distances + 1e-8)  # (b,)
        return distances

    def compute_covariance_regularization(self, z_bisim, next_z_bisim,
                                          var_target: float = 1.0,
                                          eps: float = 1e-6):
        """
        Covariance regularization.
        Operates on flattened bisim encodings (p * d) features per sample
        Args:
            z_bisim:        (b, t, p, d)
            next_z_bisim:   (b, t, p, d)
            var_target:     target variance for diagonal
            eps:            numerical stability
            offdiag_coef:   weight for off-diagonal penalty
            diag_coef:      weight for diagonal target penalty

        Returns:
            cov_reg: (b,) tensor broadcast per batch element
        """
        assert z_bisim.dim() == 4 and next_z_bisim.dim() == 4, \
            f"expected (b,t,p,d); got {z_bisim.shape} and {next_z_bisim.shape}"

        b, t, p, d = z_bisim.shape
        feature_dim = p * d

        # (b, t, p, d) -> (b, t, p*d)
        z_flat = z_bisim.reshape(b, t, feature_dim)
        next_z_flat = next_z_bisim.reshape(b, t, feature_dim)

        # stack current and next along batch axis, then flatten batch and time
        # (2*b, t, p*d) -> (2*b*t, p*d)
        Z = torch.cat([z_flat, next_z_flat], dim=0).reshape(-1, feature_dim)

        # center the data
        Zc = Z - Z.mean(dim=0, keepdim=True)

        # compute covariance matrix (feature_dim × feature_dim)
        N = Zc.shape[0]
        denom = max(N - 1, 1)
        C = (Zc.T @ Zc) / denom
        C = C + eps * torch.eye(feature_dim, device=C.device, dtype=C.dtype)

        # loss terms
        diag = torch.diag(C)  # (feature_dim,)
        diag_loss = (diag - var_target).pow(2).mean()

        # Off-diagonal penalty: ||C||_F^2 - ||diag(C)||_2^2, normalized by count
        frob2 = (C * C).sum()
        diag2 = (diag * diag).sum()
        offdiag_sum = frob2 - diag2
        offdiag_norm = feature_dim * (feature_dim - 1)
        offdiag_loss = offdiag_sum / max(offdiag_norm, 1)

        cov_reg = offdiag_loss + diag_loss

        # Return per-batch scalar (broadcast) to match your loss plumbing
        return cov_reg.expand(b)

    def var_loss(self, z_bisim, var_target=0.1, epsilon=0):
        """
        Calculate variance loss (core)
        input: z_bisim: (b, t, num_patches, patch_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t)
        """
        # (b, t, num_patches, patch_dim) -> (b, t, num_patches*patch_dim)
        b, t, num_patches, patch_dim = z_bisim.shape
        z_flat = z_bisim.reshape(b, t, num_patches * patch_dim)

        # Compute variance
        var = z_flat.var(dim=0)  # (t, num_patches*patch_dim)

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

        return loss.mean(dim=1)  # reduce the dimension to (t)

    def cal_pca(self, z_bisim):
        # (B, T, num_patches, patch_dim) -> (B*T, num_patches*patch_dim)
        B, T, n_patches, patch_dim = z_bisim.shape
        z_flat = z_bisim.reshape(B, T, n_patches * patch_dim)
        Z = z_flat.reshape(B * T, n_patches * patch_dim)

        # Center data
        Z_centered = Z - Z.mean(dim=0, keepdim=True)

        # PCA via SVD
        U, S, Vt = torch.linalg.svd(Z_centered, full_matrices=False) # check  Vt or V
        V = Vt.T  # (n_patches*patch_dim, n_patches*patch_dim)

        self.PCAMatrix = V.detach()
        self.PCA_Calced = True

    def pca_var_loss(self, z_bisim, target_first=0.01, target_rest=2.0, num_pcs=10):
        """
        PCA variance loss:
        - 1st PC variance -> target_first
        - Next (num_pcs-1) PCs variance -> target_rest
        - Remaining PCs are unconstrained

        Args:
            z_bisim: Tensor (B, T, num_patches, patch_dim)
            target_first: variance target for the first PC
            target_rest: variance target for PCs 2..num_pcs
            num_pcs: number of PCs to regularize
        Returns:
            scalar loss
        """
        # (B, T, num_patches, patch_dim) -> (B*T, num_patches*patch_dim)
        B, T, n_patches, patch_dim = z_bisim.shape
        z_flat = z_bisim.reshape(B, T, n_patches * patch_dim)
        Z = z_flat.reshape(B * T, n_patches * patch_dim)

        # Center data
        Z_centered = Z - Z.mean(dim=0, keepdim=True)

        if not self.PCA_Calced:
            self.cal_pca(z_bisim)
        V = self.PCAMatrix
        num_pcs = min(num_pcs, V.shape[1])
        V_10 = V[:, :num_pcs]  # (n_patches*patch_dim, num_pcs)

        # Project to PCA coords
        Z_proj = Z_centered @ V_10  # (B*T, num_pcs)
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
        input: z_bisim: (b, t, num_patches, patch_dim)
        next_z_bisim: (b, t, num_patches, patch_dim)
        var_target: variance parameter
        epsilon: variance parameter
        output: var_loss: (t)
        """

        T_Plus_1_z_bisim = torch.cat([z_bisim, next_z_bisim], dim=0)

        # calculate the loss
        loss = self.var_loss(T_Plus_1_z_bisim, var_target, epsilon)
        #loss = self.pca_var_loss(T_Plus_1_z_bisim, target_first, var_target, num_pcs)

        return loss  # dimension=(T+1), or memory sample+1

    def calc_PCAVar_loss(self, z_bisim, next_z_bisim, target_first=0.01, var_target=0.1, num_pcs=10):

        """
        Calculate PCA variance loss with memory buffer
        input: z_bisim: (b, t, num_patches, patch_dim)
        next_z_bisim: (b, t, num_patches, patch_dim)
        target_first: target variance for first PC
        var_target: target variance for other PCs
        num_pcs: number of principal components to regularize
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
        input: z_bisim, z_bisim2: (b, t, num_patches, patch_dim)
               reward, reward2: (b, t, 1)
               next_z_bisim, next_z_bisim2: (b, t, num_patches, patch_dim)
        output: bisim_loss: (b, t)
        """
        # Compute L2 distance per (b, t) by reducing over patches and patch_dim
        b, t, p, d = z_bisim.shape
        z1_flat = z_bisim.reshape(b, t, p * d)  # (b, t, D)
        z2_flat = z_bisim2.reshape(b, t, p * d)  # (b, t, D)

        z_dist = F.smooth_l1_loss(z1_flat, z2_flat, reduction="none").sum(dim=-1)

        # Compute distance between rewards
        r_dist = torch.sum(F.smooth_l1_loss(reward, reward2, reduction="none"), dim=-1)

        # Compute transition distance between next states
        transition_dist = self.compute_transition_distance(next_z_bisim, next_z_bisim2)  # (b,)

        # Check for NaN or Inf in transition_dist
        if torch.isnan(transition_dist).any() or torch.isinf(transition_dist).any():
            print("WARNING: NaN or Inf values detected in transition_dist!")
            print(f"transition_dist shape: {transition_dist.shape}")
            print(
                f"transition_dist stats: mean={transition_dist.mean().item():.6f}, std={transition_dist.std().item():.6f}, min={transition_dist.min().item():.6f}, max={transition_dist.max().item():.6f}")
            # Replace with safe values
            transition_dist = torch.ones_like(transition_dist)

        # Expand to match time dimension of r_dist
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
