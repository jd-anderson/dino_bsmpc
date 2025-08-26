import numpy as np
import torch
import torch.nn as nn
import json


def create_objective_fn(alpha, base, mode="last", use_bisim=False, bisim_weight=1.0, planning_space="original",
                        wm=None, detailed_log_filename="objective_log.json"):
    """
    Loss calculated on the last pred frame.
    Args:
        alpha: int
        base: int. only used for objective_fn_all
        use_bisim: bool. whether to use bisimulation metrics
        bisim_weight: float. weight of bisimulation loss (only used when planning_space=original)
        planning_space: str. 'original' for DINOv2 + weighted bisim loss, 'bisim' for planning directly in bisim space
        wm: world model instance. needed for bisimulation metrics
        detailed_log_filename: str. filename for detailed objective logging
    Returns:
        loss: tensor (B, )
    """
    metric = nn.MSELoss(reduction="none")
    
    def log_objective(data):
        """Log objective function details"""
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
        with open(detailed_log_filename, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

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
        
        # Log objective computation
        log_objective({
            "event": "objective_fn_last",
            "loss_visual": loss_visual.tolist(),
            "loss_proprio": loss_proprio.tolist(),
            "alpha": alpha,
            "final_loss": loss.tolist(),
            "z_obs_pred_shapes": {k: list(v.shape) for k, v in z_obs_pred.items()},
            "z_obs_tgt_shapes": {k: list(v.shape) for k, v in z_obs_tgt.items()},
        })
        
        return loss

    def objective_fn_last_bisim_original(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on the last pred frame using original approach (DINOv2 + weighted bisim loss).
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        # First compute the standard loss in DINOv2 space
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

            # Compute bisimulation embeddings and distance
            bisim_pred = encode_bisim(
                {"visual": z_obs_pred["visual"][:, -1:], "proprio": z_obs_pred["proprio"][:, -1:]})
            bisim_tgt = encode_bisim({"visual": z_obs_tgt["visual"], "proprio": z_obs_tgt["proprio"]})

            # Calculate L2 distance in bisimulation space - ensure it returns a scalar per batch element
            if len(bisim_pred.shape) > 2:  # If it has more than 2 dimensions [batch, timestep, ...]
                bisim_loss = torch.norm(bisim_pred - bisim_tgt, dim=tuple(range(2, bisim_pred.ndim)))
                bisim_loss = bisim_loss.mean(dim=1)  # Mean across timesteps if multiple
            else:
                bisim_loss = torch.norm(bisim_pred - bisim_tgt, dim=-1)  # Shape: [batch]

            # Combine losses
            loss = std_loss + bisim_weight * bisim_loss
            
            # Log objective computation with bisimulation
            log_objective({
                "event": "objective_fn_last_bisim_original",
                "loss_visual": loss_visual.tolist(),
                "loss_proprio": loss_proprio.tolist(),
                "alpha": alpha,
                "std_loss": std_loss.tolist(),
                "bisim_pred_shape": list(bisim_pred.shape),
                "bisim_tgt_shape": list(bisim_tgt.shape),
                "bisim_loss": bisim_loss.tolist(),
                "bisim_weight": bisim_weight,
                "final_loss": loss.tolist(),
                "z_obs_pred_shapes": {k: list(v.shape) for k, v in z_obs_pred.items()},
                "z_obs_tgt_shapes": {k: list(v.shape) for k, v in z_obs_tgt.items()},
            })
            return loss
        else:
            # Fall back to standard loss if bisimulation isn't available
            log_objective({
                "event": "objective_fn_last_bisim_original_fallback",
                "loss_visual": loss_visual.tolist(),
                "loss_proprio": loss_proprio.tolist(),
                "alpha": alpha,
                "final_loss": std_loss.tolist(),
                "reason": "no_bisimulation_available",
            })
            return std_loss

    def objective_fn_last_bisim_space(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on the last pred frame using bisimulation embeddings instead of DINOv2.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        # Only use bisimulation if the world model is provided and has bisimulation
        if wm is not None and hasattr(wm, 'has_bisim') and wm.has_bisim:
            # Use the world model's encode_bisim function to replace DINOv2 embeddings
            encode_bisim = wm.encode_bisim

            # Get bisimulation embeddings instead of using DINOv2 embeddings directly
            bisim_pred_visual = encode_bisim(
                {"visual": z_obs_pred["visual"][:, -1:], "proprio": z_obs_pred["proprio"][:, -1:]})
            bisim_tgt_visual = encode_bisim({"visual": z_obs_tgt["visual"], "proprio": z_obs_tgt["proprio"]})

            # Use the same loss calculation logic as the standard approach, but with bisimulation embeddings
            loss_visual = metric(bisim_pred_visual, bisim_tgt_visual).mean(
                dim=tuple(range(1, bisim_pred_visual.ndim))
            )
            loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
                dim=tuple(range(1, z_obs_pred["proprio"].ndim))
            )
            loss = loss_visual + alpha * loss_proprio
            
            # Log objective computation with bisimulation space
            log_objective({
                "event": "objective_fn_last_bisim_space",
                "bisim_pred_visual_shape": list(bisim_pred_visual.shape),
                "bisim_tgt_visual_shape": list(bisim_tgt_visual.shape),
                "loss_visual": loss_visual.tolist(),
                "loss_proprio": loss_proprio.tolist(),
                "alpha": alpha,
                "final_loss": loss.tolist(),
                "z_obs_pred_shapes": {k: list(v.shape) for k, v in z_obs_pred.items()},
                "z_obs_tgt_shapes": {k: list(v.shape) for k, v in z_obs_tgt.items()},
            })
            
            return loss
        else:
            # Fall back to standard loss if bisimulation isn't available
            loss_visual = metric(z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]).mean(
                dim=tuple(range(1, z_obs_pred["visual"].ndim))
            )
            loss_proprio = metric(z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]).mean(
                dim=tuple(range(1, z_obs_pred["proprio"].ndim))
            )
            loss = loss_visual + alpha * loss_proprio
            
            log_objective({
                "event": "objective_fn_last_bisim_space_fallback",
                "loss_visual": loss_visual.tolist(),
                "loss_proprio": loss_proprio.tolist(),
                "alpha": alpha,
                "final_loss": loss.tolist(),
                "reason": "no_bisimulation_available",
            })
            
            return loss

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
        
        log_objective({
            "event": "objective_fn_all",
            "coeffs": coeffs.tolist(),
            "loss_visual": loss_visual.tolist(),
            "loss_proprio": loss_proprio.tolist(),
            "alpha": alpha,
            "final_loss": loss.tolist(),
            "z_obs_pred_shapes": {k: list(v.shape) for k, v in z_obs_pred.items()},
            "z_obs_tgt_shapes": {k: list(v.shape) for k, v in z_obs_tgt.items()},
        })
        
        return loss

    def objective_fn_all_bisim_original(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames using original approach (DINOv2 + weighted bisim loss).
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        # First compute the standard loss in DINOv2 space
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

            # Calculate L2 distance in bisimulation space - ensure it returns a scalar per batch element
            if len(bisim_pred.shape) > 2:  # If it has more than 2 dimensions [batch, timestep, ...]
                bisim_dist = torch.norm(bisim_pred - bisim_tgt, dim=tuple(range(2, bisim_pred.ndim)))
            else:
                bisim_dist = torch.norm(bisim_pred - bisim_tgt, dim=-1)  # Shape: [batch, timesteps]

            # Apply coefficients
            bisim_loss = (bisim_dist * coeffs).mean(dim=1)

            # Combine losses
            loss = std_loss + bisim_weight * bisim_loss
            
            log_objective({
                "event": "objective_fn_all_bisim_original",
                "coeffs": coeffs.tolist(),
                "loss_visual": loss_visual.tolist(),
                "loss_proprio": loss_proprio.tolist(),
                "alpha": alpha,
                "std_loss": std_loss.tolist(),
                "bisim_pred_shape": list(bisim_pred.shape),
                "bisim_tgt_shape": list(bisim_tgt.shape),
                "bisim_loss": bisim_loss.tolist(),
                "bisim_weight": bisim_weight,
                "final_loss": loss.tolist(),
                "z_obs_pred_shapes": {k: list(v.shape) for k, v in z_obs_pred.items()},
                "z_obs_tgt_shapes": {k: list(v.shape) for k, v in z_obs_tgt.items()},
            })
            
            return loss
        else:
            # Fall back to standard loss if bisimulation isn't available
            log_objective({
                "event": "objective_fn_all_bisim_original_fallback",
                "coeffs": coeffs.tolist(),
                "loss_visual": loss_visual.tolist(),
                "loss_proprio": loss_proprio.tolist(),
                "alpha": alpha,
                "final_loss": std_loss.tolist(),
                "reason": "no_bisimulation_available",
            })
            return std_loss

    def objective_fn_all_bisim_space(z_obs_pred, z_obs_tgt):
        """
        Loss calculated on all pred frames using bisimulation embeddings instead of DINOv2.
        Args:
            z_obs_pred: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
            z_obs_tgt: dict, {'visual': (B, T, *D_visual), 'proprio': (B, T, *D_proprio)}
        Returns:
            loss: tensor (B, )
        """
        # Only use bisimulation if the world model is provided and has bisimulation
        if wm is not None and hasattr(wm, 'has_bisim') and wm.has_bisim:
            # Use the world model's encode_bisim function to replace DINOv2 embeddings
            encode_bisim = wm.encode_bisim

            # Compute bisimulation embeddings for each timestep (replacing visual embeddings)
            bisim_pred_visual_all = []
            bisim_tgt_visual_all = []

            for t in range(z_obs_pred["visual"].shape[1]):
                bisim_pred_visual_t = encode_bisim({
                    "visual": z_obs_pred["visual"][:, t:t + 1],
                    "proprio": z_obs_pred["proprio"][:, t:t + 1]
                })
                bisim_tgt_visual_t = encode_bisim({
                    "visual": z_obs_tgt["visual"],
                    "proprio": z_obs_tgt["proprio"]
                })
                bisim_pred_visual_all.append(bisim_pred_visual_t)
                bisim_tgt_visual_all.append(bisim_tgt_visual_t)

            # Stack tensors
            bisim_pred_visual = torch.stack(bisim_pred_visual_all, dim=1)
            bisim_tgt_visual = torch.stack(bisim_tgt_visual_all, dim=1)

            # Use the same loss calculation logic as the standard approach
            coeffs = np.array(
                [base ** i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
            )
            coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)

            # Calculate loss for visual (using bisimulation embeddings) and proprio separately
            loss_visual = metric(bisim_pred_visual, bisim_tgt_visual).mean(
                dim=tuple(range(2, bisim_pred_visual.ndim))
            )
            loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
                dim=tuple(range(2, z_obs_pred["proprio"].ndim))
            )

            # Apply coefficients and combine
            loss_visual = (loss_visual * coeffs).mean(dim=1)
            loss_proprio = (loss_proprio * coeffs).mean(dim=1)
            loss = loss_visual + alpha * loss_proprio
            
            log_objective({
                "event": "objective_fn_all_bisim_space",
                "coeffs": coeffs.tolist(),
                "bisim_pred_visual_shape": list(bisim_pred_visual.shape),
                "bisim_tgt_visual_shape": list(bisim_tgt_visual.shape),
                "loss_visual": loss_visual.tolist(),
                "loss_proprio": loss_proprio.tolist(),
                "alpha": alpha,
                "final_loss": loss.tolist(),
                "z_obs_pred_shapes": {k: list(v.shape) for k, v in z_obs_pred.items()},
                "z_obs_tgt_shapes": {k: list(v.shape) for k, v in z_obs_tgt.items()},
            })
            
            return loss
        else:
            # Fall back to standard loss if bisimulation isn't available
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
            
            log_objective({
                "event": "objective_fn_all_bisim_space_fallback",
                "coeffs": coeffs.tolist(),
                "loss_visual": loss_visual.tolist(),
                "loss_proprio": loss_proprio.tolist(),
                "alpha": alpha,
                "final_loss": loss.tolist(),
                "reason": "no_bisimulation_available",
            })
            
            return loss

    if mode == "last":
        if use_bisim:
            if planning_space == "bisim":
                return objective_fn_last_bisim_space
            else:  # planning_space == "original"
                return objective_fn_last_bisim_original
        else:
            return objective_fn_last
    elif mode == "all":
        if use_bisim:
            if planning_space == "bisim":
                return objective_fn_all_bisim_space
            else:  # planning_space == "original"
                return objective_fn_all_bisim_original
        else:
            return objective_fn_all
    else:
        raise NotImplementedError
