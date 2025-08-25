import torch
import numpy as np
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
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)

        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        n_evals = mu.shape[0]

        for i in range(self.opt_steps):
            # optimize individual instances
            losses = []
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
                with torch.no_grad():
                    # Check if the model has bisimulation capabilities
                    has_bisim = hasattr(self.wm, 'has_bisim') and self.wm.has_bisim
                    if i == 0 and traj == 0:  # Only log on first iteration for first trajectory
                        pass
                    
                    # Run rollout
                    rollout_result = self.wm.rollout(
                        obs_0=cur_trans_obs_0,
                        act=action,
                    )
                    
                    # Handle both cases: with or without bisimulation
                    if len(rollout_result) == 3:
                        i_z_obses, i_zs, bisim_z = rollout_result
                        # if i == 0 or (i % 10 == 0 and traj == 0):
                            # print(f"\nCEM PREDICTION: Iteration {i}, using bisimulation embeddings")
                            # print(f"CEM PREDICTION: Bisimulation embeddings shape: {bisim_z.shape}, dim={bisim_z.shape[-1]}")
                            # print(f"CEM PREDICTION: Visual embeddings shape: {i_z_obses['visual'].shape}")
                            # print(f"CEM PREDICTION: Proprio embeddings shape: {i_z_obses['proprio'].shape}")
                            # if bisim_z.shape[-1] != 64:
                            #     print(f"WARNING: Unexpected bisimulation dimension: {bisim_z.shape[-1]}, expected 64")
                    else:
                        i_z_obses, i_zs = rollout_result
                        # if i == 0 or (i % 10 == 0 and traj == 0):
                        #     print(f"\nCEM PREDICTION: Iteration {i}, not using bisimulation")
                        #     print(f"CEM PREDICTION: Visual embeddings shape: {i_z_obses['visual'].shape}")
                        #     print(f"CEM PREDICTION: Proprio embeddings shape: {i_z_obses['proprio'].shape}")

                # Print right before computing loss
                # if (traj == 0 and i == 0) or (traj == 0 and i % 10 == 0):
                #     print(f"CEM PREDICTION: About to compute loss with predicted obs -> goal")
                #     print(f"CEM PREDICTION: Current trajectory: {traj+1}/{n_evals}, iteration: {i+1}/{self.opt_steps}")

                # Compute loss
                if 'bisim' in locals() and bisim_z is not None:
                    pred_dict = {"bisim": bisim_z}
                    tgt_bisim = self.wm.encode_bisim(cur_z_obs_g)
                    tgt_dict = {"bisim": tgt_bisim}
                    loss = self.objective_fn(pred_dict, tgt_dict)
                else:
                    loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                
                # Ensure loss has the right dimensionality (num_samples,)
                if loss.ndim > 1:
                    loss = loss.mean(dim=tuple(range(1, loss.ndim)))
                
                # Get top actions
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                top_loss = loss[topk_idx[0]].item()
                losses.append(top_loss)
                
                # Update distribution parameters
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)

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
                if np.all(successes):
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid
