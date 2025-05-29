import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base, mode="last", use_bisim=False, bisim_weight=1.0, wm=None):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
        use_bisim: bool. whether to use bisimulation metrics
        bisim_weight: float. weight of bisimulation loss
        wm: world model instance. needed for bisimulation metrics
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        """
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_last_bisim(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on the last pred frame using bisimulation metrics.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        # First compute the standard loss
        loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
            dim=tuple(range(1, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(1, z_obs_pred["proprio"].ndim))
        )
        std_loss = loss_visual + alpha * loss_proprio

        # Only use bisimulation if the world model is provided and has bisimulation
        if wm is not None and hasattr(wm, 'has_bisim') and wm.has_bisim:
            # Use the world model's encode_bisim function
            encode_bisim = wm.encode_bisim
            # print(f"BISIM PLANNING: Using bisimulation loss with weight {bisim_weight} (mode=last)")

            # Compute bisimulation embeddings and distance
            bisim_pred = encode_bisim({"visual": z_obs_pred["visual"][:, -1:], "proprio": z_obs_pred["proprio"][:, -1:]})
            bisim_tgt = encode_bisim({"visual": z_obs_tgt["visual"], "proprio": z_obs_tgt["proprio"]})
            
            # print(f"BISIM PLANNING: Using bisimulation embeddings of dimension {bisim_pred.shape[-1]}")
            # print(f"BISIM PLANNING: Predicted state shape={bisim_pred.shape}, target state shape={bisim_tgt.shape}")

            # Calculate L2 distance in bisimulation space - ensure it returns a scalar per batch element
            if len(bisim_pred.shape) > 2:  # If it has more than 2 dimensions [batch, timestep, ...]
                bisim_loss = torch.norm(bisim_pred - bisim_tgt, dim=tuple(range(2, bisim_pred.ndim)))
                bisim_loss = bisim_loss.mean(dim=1)  # Mean across timesteps if multiple
            else:
                bisim_loss = torch.norm(bisim_pred - bisim_tgt, dim=-1)  # Shape: [batch]

            # Log loss statistics
            # print(f"BISIM PLANNING: Loss stats - std_loss min/max/mean: {std_loss.min().item():.4f}/{std_loss.max().item():.4f}/{std_loss.mean().item():.4f}")
            # print(f"BISIM PLANNING: Loss stats - bisim_loss min/max/mean: {bisim_loss.min().item():.4f}/{bisim_loss.max().item():.4f}/{bisim_loss.mean().item():.4f}")

            # Combine losses
            loss = std_loss + bisim_weight * bisim_loss
            return loss
        else:
            # Fall back to standard loss if bisimulation isn't available
            return std_loss

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        coeffs = np.array(
            [base ** i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        loss = loss_visual + alpha * loss_proprio
        return loss

    def objective_fn_all_bisim(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames using bisimulation metrics.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        # First compute the standard loss
        coeffs = np.array(
            [base ** i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)
        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        std_loss = loss_visual + alpha * loss_proprio

        # Only use bisimulation if the world model is provided and has bisimulation
        if wm is not None and hasattr(wm, 'has_bisim') and wm.has_bisim:
            # Use the world model's encode_bisim function
            encode_bisim = wm.encode_bisim
            # print(f"BISIM PLANNING: Using bisimulation loss with weight {bisim_weight} (mode=all)")

            # Compute bisimulation embeddings for each timestep
            bisim_pred_all = []
            bisim_tgt_all = []

            for t in range(z_obs_pred["visual"].shape[1]):
                bisim_pred_t = encode_bisim({
                    "visual": z_obs_pred["visual"][:, t:t + 1],
                    "proprio": z_obs_pred["proprio"][:, t:t + 1]
                })
                bisim_tgt_t = encode_bisim({
                    "visual": z_obs_tgt["visual"],
                    "proprio": z_obs_tgt["proprio"]
                })
                bisim_pred_all.append(bisim_pred_t)
                bisim_tgt_all.append(bisim_tgt_t)

            # Stack tensors
            bisim_pred = torch.stack(bisim_pred_all, dim=1)
            bisim_tgt = torch.stack(bisim_tgt_all, dim=1)
            
            # print(f"BISIM PLANNING: Stacked predictions shape: {bisim_pred.shape}")

            # Calculate L2 distance in bisimulation space - ensure it returns a scalar per batch element
            if len(bisim_pred.shape) > 2:  # If it has more than 2 dimensions [batch, timestep, ...]
                bisim_dist = torch.norm(bisim_pred - bisim_tgt, dim=tuple(range(2, bisim_pred.ndim)))
            else:
                bisim_dist = torch.norm(bisim_pred - bisim_tgt, dim=-1)  # Shape: [batch, timesteps]

            # Apply coefficients
            bisim_loss = (bisim_dist * coeffs).mean(dim=1)
            
            # Log loss statistics
            # print(f"BISIM PLANNING: Loss stats - std_loss min/max/mean: {std_loss.min().item():.4f}/{std_loss.max().item():.4f}/{std_loss.mean().item():.4f}")
            # print(f"BISIM PLANNING: Loss stats - bisim_loss min/max/mean: {bisim_loss.min().item():.4f}/{bisim_loss.max().item():.4f}/{bisim_loss.mean().item():.4f}")

            # Combine losses
            loss = std_loss + bisim_weight * bisim_loss
            return loss
        else:
            # Fall back to standard loss if bisimulation isn't available
            return std_loss

    if mode == "last":
        return objective_fn_last_bisim if use_bisim else objective_fn_last
    elif mode == "all":
        return objective_fn_all_bisim if use_bisim else objective_fn_all
    else:
        raise NotImplementedError
