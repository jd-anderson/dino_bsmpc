import torch
import numpy as np
import json
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device


class CEMPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        detailed_log_filename="detailed_planning_log.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.detailed_log_filename = detailed_log_filename
        
        # Initialize detailed logging
        self.detailed_logs = []
        self.planning_session_id = f"plan_{np.random.randint(10000, 99999)}"

    def log_detailed(self, data):
        """Log detailed planning information"""
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
            "session_id": self.planning_session_id,
            **serializable_data
        }
        self.detailed_logs.append(log_entry)
        
        # Also write to file immediately
        with open(self.detailed_log_filename, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        
        # Log initialization
        self.log_detailed({
            "event": "init_mu_sigma",
            "n_evals": n_evals,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "var_scale": self.var_scale,
            "mu_shape": list(mu.shape),
            "sigma_shape": list(sigma.shape),
            "has_initial_actions": actions is not None,
            "initial_actions_shape": list(actions.shape) if actions is not None else None,
        })
        
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        # Log planning start
        self.log_detailed({
            "event": "planning_start",
            "obs_0_keys": list(obs_0.keys()),
            "obs_g_keys": list(obs_g.keys()),
            "obs_0_shapes": {k: list(v.shape) for k, v in obs_0.items()},
            "obs_g_shapes": {k: list(v.shape) for k, v in obs_g.items()},
            "has_bisim": hasattr(self.wm, 'has_bisim') and self.wm.has_bisim,
            "bisim_latent_dim": getattr(self.wm, 'bisim_latent_dim', None),
        })
        
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        # Log goal encoding
        self.log_detailed({
            "event": "goal_encoding",
            "trans_obs_g_shapes": {k: list(v.shape) for k, v in trans_obs_g.items()},
            "z_obs_g_shapes": {k: list(v.shape) for k, v in z_obs_g.items()},
            "z_obs_g_stats": {
                k: {
                    "mean": float(v.mean().item()),
                    "std": float(v.std().item()),
                    "min": float(v.min().item()),
                    "max": float(v.max().item()),
                } for k, v in z_obs_g.items()
            }
        })

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # Log iteration start
            self.log_detailed({
                "event": "iteration_start",
                "iteration": i,
                "mu_stats": {
                    "mean": float(mu.mean().item()),
                    "std": float(mu.std().item()),
                    "min": float(mu.min().item()),
                    "max": float(mu.max().item()),
                },
                "sigma_stats": {
                    "mean": float(sigma.mean().item()),
                    "std": float(sigma.std().item()),
                    "min": float(sigma.min().item()),
                    "max": float(sigma.max().item()),
                }
            })
            
            # optimize individual instances
            losses = []
            iteration_details = []
            
            for traj in range(n_evals):
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action = (
                    torch.randn(self.num_samples, self.horizon, self.action_dim).to(
                        self.device
                    )
                    * sigma[traj]
                    + mu[traj]
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                
                # Log action sampling
                self.log_detailed({
                    "event": "action_sampling",
                    "iteration": i,
                    "trajectory": traj,
                    "action_stats": {
                        "mean": float(action.mean().item()),
                        "std": float(action.std().item()),
                        "min": float(action.min().item()),
                        "max": float(action.max().item()),
                    },
                    "action_shape": list(action.shape),
                })
                
                with torch.no_grad():
                    # Check if the model has bisimulation capabilities
                    has_bisim = hasattr(self.wm, 'has_bisim') and self.wm.has_bisim
                    
                    # Run rollout
                    rollout_result = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )
                    
                    # Handle both cases: with or without bisimulation
                    if len(rollout_result) == 3:
                        i_z_obses, i_zs, bisim_z = rollout_result
                        
                        # Log bisimulation embeddings
                        self.log_detailed({
                            "event": "bisimulation_embeddings",
                            "iteration": i,
                            "trajectory": traj,
                            "bisim_z_shape": list(bisim_z.shape),
                            "bisim_z_stats": {
                                "mean": float(bisim_z.mean().item()),
                                "std": float(bisim_z.std().item()),
                                "min": float(bisim_z.min().item()),
                                "max": float(bisim_z.max().item()),
                            },
                            "visual_emb_shape": list(i_z_obses['visual'].shape),
                            "proprio_emb_shape": list(i_z_obses['proprio'].shape),
                        })
                        
                    else:
                        i_z_obses, i_zs = rollout_result
                        
                        # Log regular embeddings
                        self.log_detailed({
                            "event": "regular_embeddings",
                            "iteration": i,
                            "trajectory": traj,
                            "visual_emb_shape": list(i_z_obses['visual'].shape),
                            "proprio_emb_shape": list(i_z_obses['proprio'].shape),
                        })

                # Compute loss
                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                
                # Ensure loss has the right dimensionality (num_samples,)
                if loss.ndim > 1:
                    loss = loss.mean(dim=tuple(range(1, loss.ndim)))
                
                # Log loss computation
                self.log_detailed({
                    "event": "loss_computation",
                    "iteration": i,
                    "trajectory": traj,
                    "loss_shape": list(loss.shape),
                    "loss_stats": {
                        "mean": float(loss.mean().item()),
                        "std": float(loss.std().item()),
                        "min": float(loss.min().item()),
                        "max": float(loss.max().item()),
                    },
                    "loss_values": loss.tolist()[:10],  # Log first 10 values
                })
                
                # Get top actions
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                top_loss = loss[topk_idx[0]].item()
                losses.append(top_loss)
                
                # Log top actions selection
                self.log_detailed({
                    "event": "top_actions_selection",
                    "iteration": i,
                    "trajectory": traj,
                    "topk": self.topk,
                    "top_loss": top_loss,
                    "topk_indices": topk_idx.tolist(),
                    "topk_action_stats": {
                        "mean": float(topk_action.mean().item()),
                        "std": float(topk_action.std().item()),
                        "min": float(topk_action.min().item()),
                        "max": float(topk_action.max().item()),
                    },
                })
                
                # Update distribution parameters
                old_mu = mu[traj].clone()
                old_sigma = sigma[traj].clone()
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)
                
                # Log distribution update
                self.log_detailed({
                    "event": "distribution_update",
                    "iteration": i,
                    "trajectory": traj,
                    "mu_change": {
                        "mean": float((mu[traj] - old_mu).mean().item()),
                        "std": float((mu[traj] - old_mu).std().item()),
                    },
                    "sigma_change": {
                        "mean": float((sigma[traj] - old_sigma).mean().item()),
                        "std": float((sigma[traj] - old_sigma).std().item()),
                    },
                })
                
                # Store trajectory details
                trajectory_detail = {
                    "trajectory": traj,
                    "loss": top_loss,
                    "topk_indices": topk_idx.tolist(),
                    "action_stats": {
                        "mean": float(action.mean().item()),
                        "std": float(action.std().item()),
                    },
                    "topk_action_stats": {
                        "mean": float(topk_action.mean().item()),
                        "std": float(topk_action.std().item()),
                    },
                }
                
                # Add bisimulation info if available
                if len(rollout_result) == 3:
                    trajectory_detail["bisim_z_stats"] = {
                        "mean": float(bisim_z.mean().item()),
                        "std": float(bisim_z.std().item()),
                    }
                
                iteration_details.append(trajectory_detail)

            # Log iteration summary
            self.log_detailed({
                "event": "iteration_summary",
                "iteration": i,
                "mean_loss": np.mean(losses),
                "loss_std": np.std(losses),
                "min_loss": np.min(losses),
                "max_loss": np.max(losses),
                "trajectory_details": iteration_details,
            })

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": np.mean(losses), "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                
                # Log evaluation results
                self.log_detailed({
                    "event": "evaluation",
                    "iteration": i,
                    "successes": successes.tolist(),
                    "success_rate": float(np.mean(successes)),
                    "evaluation_logs": logs,
                })
                
                if np.all(successes):
                    self.log_detailed({
                        "event": "early_termination",
                        "iteration": i,
                        "reason": "all_successes",
                    })
                    break  # terminate planning if all success

        # Log planning completion
        self.log_detailed({
            "event": "planning_completion",
            "final_mu_stats": {
                "mean": float(mu.mean().item()),
                "std": float(mu.std().item()),
                "min": float(mu.min().item()),
                "max": float(mu.max().item()),
            },
            "final_sigma_stats": {
                "mean": float(sigma.mean().item()),
                "std": float(sigma.std().item()),
                "min": float(sigma.min().item()),
                "max": float(sigma.max().item()),
            },
        })

        return mu, np.full(n_evals, np.inf)  # all actions are valid
