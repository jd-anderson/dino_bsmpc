import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
import numpy as np


class VWorldModel(nn.Module):
    def __init__(
            self,
            image_size,  # 224
            num_hist,
            num_pred,
            encoder,
            proprio_encoder,
            action_encoder,
            decoder,
            predictor,
            bisim_model=None,
            bisim_latent_dim=32,
            bisim_hidden_dim=256,
            bisim_coef=1.0,
            var_loss_coef: float = 1.0,
            PCA1_loss_target: float = 0.01,
            VC_target: float = 1.0,
            num_pcs: int = 10,
            PCAloss_epoch: int = 50,
            train_bisim=True,
            bypass_dinov2=False,
            bisim_memory_buffer_size=0,
            bisim_comparison_size=20,
            proprio_dim=0,
            action_dim=0,
            concat_dim=0,
            num_action_repeat=7,
            num_proprio_repeat=7,
            train_encoder=True,
            train_predictor=False,
            train_decoder=True,
            train_w_std_loss=True,
            train_w_reward_loss=True,
            accelerate=False,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat
        self.action_dim = action_dim * num_action_repeat
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim)  # Not used

        self.has_bisim = bisim_model is not None
        self.bypass_dinov2 = bypass_dinov2
        self._bisim_encode_logged = False

        if self.has_bisim:
            self.bisim_model = bisim_model
            self.bisim_latent_dim = bisim_latent_dim
            self.bisim_hidden_dim = bisim_hidden_dim
            self.bisim_patch_dim = bisim_latent_dim
            self.train_bisim = train_bisim
            self.bisim_coef = bisim_coef

            # memory buffer setup
            self.bisim_memory_buffer_size = bisim_memory_buffer_size
            self.bisim_comparison_size = bisim_comparison_size
            self.use_memory_buffer = bisim_memory_buffer_size > 0

            if self.use_memory_buffer:
                action_emb_dim = getattr(action_encoder, 'emb_dim', action_dim)
                # calculate num_patches
                decoder_scale = 16
                num_side_patches = image_size // decoder_scale
                num_patches = num_side_patches ** 2
                self.register_buffer('bisim_memory_states',
                                     torch.zeros(bisim_memory_buffer_size, num_hist, num_patches, self.bisim_patch_dim))
                self.register_buffer('bisim_memory_next_states',
                                     torch.zeros(bisim_memory_buffer_size, num_hist, num_patches, self.bisim_patch_dim))
                self.register_buffer('bisim_memory_actions',
                                     torch.zeros(bisim_memory_buffer_size, num_hist, action_emb_dim))
                self.register_buffer('bisim_memory_rewards', torch.zeros(bisim_memory_buffer_size, num_hist, 1))
                self.register_buffer('bisim_memory_ptr', torch.zeros(1, dtype=torch.long))
                self.register_buffer('bisim_memory_full', torch.zeros(1, dtype=torch.bool))
                print(f"Initialized bisimulation memory buffer with size {bisim_memory_buffer_size}, num_patches={num_patches}")
            else:
                print("Bisimulation memory buffer disabled (size=0)")
        else:
            self.bisim_model = None
            self.train_bisim = False
            self.bisim_coef = 0.0
            self.use_memory_buffer = False

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim  # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        # DINOv2 (vits14) needs image resizing to match decoder scale
        # SimDINOv2 (vitb16) and IBOT (vits16) use 224x224 directly without resizing
        is_dinov2 = "dino" in self.encoder.name and "simdino" not in self.encoder.name
        is_ibot = "ibot" in self.encoder.name
        if is_dinov2 and not is_ibot:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
            print(f"DINOv2 encoder: resizing images to {self.encoder_image_size}x{self.encoder_image_size}")
        else:
            self.encoder_image_size = image_size
            self.encoder_transform = lambda x: x
            print(f"Encoder {self.encoder.name}: using original image size {image_size}x{image_size}")

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

        # bisim loss params
        self.var_loss_coef = var_loss_coef
        self.PCA1_loss_target = PCA1_loss_target
        self.VC_target = VC_target
        self.num_pcs = num_pcs
        self.PCAloss_epoch = PCAloss_epoch
        self.train_w_std_loss = train_w_std_loss
        self.train_w_reward_loss = train_w_reward_loss
        self.accelerate = accelerate

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        if self.has_bisim and self.train_bisim:
            self.bisim_model.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        if self.has_bisim:
            self.bisim_model.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode_bisim(self, input_data):
        """
        Maps input to bisimulation embeddings
        input:
        - If bypass_dinov2=False: z_dino (dict): {"visual": (b, t, p, d), "proprio": (b, t, d)} - DinoV2 embeddings
        - If bypass_dinov2=True: obs (dict): {"visual": (b, t, 3, h, w), "proprio": (b, t, d)} - Raw observations
        output: z_bisim: (b, t, num_patches, patch_dim)
        """
        if self.bypass_dinov2:
            visual_obs = input_data["visual"]  # (b, t, 3, h, w)
            if hasattr(self.bisim_model, "module"):
                z_bisim = self.bisim_model.module.encode(visual_obs)
            else:
                z_bisim = self.bisim_model.encode(visual_obs)
        else:
            if hasattr(self.bisim_model, "module"):
                z_bisim = self.bisim_model.module.encode(input_data["visual"])
            else:
                z_bisim = self.bisim_model.encode(input_data["visual"])

        return z_bisim

    def update_memory_buffer(self, z_bisim, next_z_bisim, action_emb, reward):
        """
        Update the memory buffer with new samples
        """
        if not self.use_memory_buffer or not self.training:
            return

        b, t, n_patches, patch_dim = z_bisim.shape

        # store samples in memory buffer using circular buffer
        for i in range(b):
            ptr = self.bisim_memory_ptr.item()

            # store the sample
            self.bisim_memory_states[ptr] = z_bisim[i].detach()
            self.bisim_memory_next_states[ptr] = next_z_bisim[i].detach()
            self.bisim_memory_actions[ptr] = action_emb[i].detach()
            self.bisim_memory_rewards[ptr] = reward[i].detach()

            # update pointer
            ptr = (ptr + 1) % self.bisim_memory_buffer_size
            self.bisim_memory_ptr[0] = ptr

            # mark buffer as full if we've wrapped around
            if ptr == 0:
                self.bisim_memory_full[0] = True

    def sample_from_memory_buffer(self, num_samples):
        if not self.use_memory_buffer:
            return None

        # determine how many samples are available
        if self.bisim_memory_full.item():
            available_samples = self.bisim_memory_buffer_size
        else:
            available_samples = self.bisim_memory_ptr.item()

        if available_samples == 0:
            return None

        # sample random indices
        num_samples = min(num_samples, available_samples)
        indices = torch.randperm(available_samples, device=self.bisim_memory_states.device)[:num_samples]

        return {
            'states': self.bisim_memory_states[indices],
            'next_states': self.bisim_memory_next_states[indices],
            'actions': self.bisim_memory_actions[indices],
            'rewards': self.bisim_memory_rewards[indices]
        }

    def calc_bisim_loss(self, z_bisim, next_z_bisim, action_emb, epoch, reward=None, discount=0.99):
        """
        Calculate bisimulation loss
        """
        if not self.has_bisim:
            return torch.tensor(0.0, device=z_bisim.device)

        b, t, n_patches, patch_dim = z_bisim.shape
        batch_size = b

        if reward is None:
            z_bisim_flat = z_bisim.reshape(b * t, n_patches, -1)
            action_emb_flat = action_emb.reshape(b * t, -1)

            if hasattr(self.bisim_model, "module"):
                reward = self.bisim_model.module.predict_reward(z_bisim_flat, action_emb_flat)
            else:
                reward = self.bisim_model.predict_reward(z_bisim_flat, action_emb_flat)
            reward = reward.reshape(b, t, 1)

        if self.use_memory_buffer and self.training:
            # check how many samples are available in memory
            if self.bisim_memory_full.item():
                available_memory_samples = self.bisim_memory_buffer_size
            else:
                available_memory_samples = self.bisim_memory_ptr.item()

            # full current batch + memory samples
            memory_samples_needed = self.bisim_comparison_size - batch_size

            # check if we have enough memory samples
            if available_memory_samples >= memory_samples_needed:
                # cross-batch comparison
                memory_data = self.sample_from_memory_buffer(memory_samples_needed)
                memory_states = memory_data['states'].to(z_bisim.device)
                memory_next_states = memory_data['next_states'].to(z_bisim.device)
                memory_rewards = memory_data['rewards'].to(z_bisim.device)

                z_bisim_combined = torch.cat([z_bisim, memory_states], dim=0)
                next_z_bisim_combined = torch.cat([next_z_bisim, memory_next_states], dim=0)
                reward_combined = torch.cat([reward, memory_rewards], dim=0)

                perm = torch.randperm(self.bisim_comparison_size, device=z_bisim.device)
                z_bisim2 = z_bisim_combined[perm]
                next_z_bisim2 = next_z_bisim_combined[perm]
                reward2 = reward_combined[perm]

                if hasattr(self.bisim_model, "module"):
                    bisim_loss, z_dist, r_dist, transition_dist, var_loss, cov_reg = self.bisim_model.module.calc_bisim_loss(
                        z_bisim_combined, z_bisim2, reward_combined, reward2,
                        next_z_bisim_combined, next_z_bisim2, epoch, discount, self.train_w_reward_loss,
                        self.var_loss_coef,
                        self.PCA1_loss_target, self.VC_target, self.num_pcs, self.PCAloss_epoch
                    )
                else:
                    bisim_loss, z_dist, r_dist, transition_dist, var_loss, cov_reg = self.bisim_model.calc_bisim_loss(
                        z_bisim_combined, z_bisim2, reward_combined, reward2,
                        next_z_bisim_combined, next_z_bisim2, epoch, discount, self.train_w_reward_loss,
                        self.var_loss_coef,
                        self.PCA1_loss_target, self.VC_target, self.num_pcs, self.PCAloss_epoch
                    )

                bisim_loss = bisim_loss[:batch_size]
            else:
                # fallback to batch_size comparison
                perm = torch.randperm(batch_size, device=z_bisim.device)
                z_bisim2 = z_bisim[perm]
                next_z_bisim2 = next_z_bisim[perm]
                reward2 = reward[perm]

                if hasattr(self.bisim_model, "module"):
                    bisim_loss, z_dist, r_dist, transition_dist, var_loss, cov_reg = self.bisim_model.module.calc_bisim_loss(
                        z_bisim, z_bisim2, reward, reward2,
                        next_z_bisim, next_z_bisim2, epoch, discount, self.train_w_reward_loss, self.var_loss_coef,
                        self.PCA1_loss_target, self.VC_target, self.num_pcs, self.PCAloss_epoch
                    )
                else:
                    bisim_loss, z_dist, r_dist, transition_dist, var_loss, cov_reg = self.bisim_model.calc_bisim_loss(
                        z_bisim, z_bisim2, reward, reward2,
                        next_z_bisim, next_z_bisim2, epoch, discount, self.train_w_reward_loss, self.var_loss_coef,
                        self.PCA1_loss_target, self.VC_target, self.num_pcs, self.PCAloss_epoch
                    )
        else:
            # memory buffer disabled or eval mode
            perm = torch.randperm(batch_size, device=z_bisim.device)
            z_bisim2 = z_bisim[perm]
            next_z_bisim2 = next_z_bisim[perm]
            reward2 = reward[perm]

            if hasattr(self.bisim_model, "module"):
                bisim_loss, z_dist, r_dist, transition_dist, var_loss, cov_reg = self.bisim_model.module.calc_bisim_loss(
                    z_bisim, z_bisim2, reward, reward2,
                    next_z_bisim, next_z_bisim2, epoch, discount, self.train_w_reward_loss, self.var_loss_coef,
                    self.PCA1_loss_target, self.VC_target, self.num_pcs, self.PCAloss_epoch
                )
            else:
                bisim_loss, z_dist, r_dist, transition_dist, var_loss, cov_reg = self.bisim_model.calc_bisim_loss(
                    z_bisim, z_bisim2, reward, reward2,
                    next_z_bisim, next_z_bisim2, epoch, discount, self.train_w_reward_loss, self.var_loss_coef,
                    self.PCA1_loss_target, self.VC_target, self.num_pcs, self.PCAloss_epoch
                )

        if self.training:
            self.update_memory_buffer(z_bisim, next_z_bisim, action_emb, reward)

        return bisim_loss, z_dist, r_dist, transition_dist, var_loss, cov_reg

    def encode(self, obs, act):
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size)
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)

        if self.has_bisim:
            if self.bypass_dinov2:
                raw_obs = {"visual": obs["visual"], "proprio": z_dct["proprio"]}
                z_dct["visual"] = self.encode_bisim(raw_obs)
            else:
                z_dct["visual"] = self.encode_bisim(z_dct)

        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z = torch.cat(
                [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2  # add as an extra token
            )  # (b, num_frames, num_patches + 2, dim)
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z = torch.cat(
                [z_dct['visual'], proprio_repeated, act_repeated], dim=3
            )  # (b, num_frames, num_patches, dim + action_dim)
        return z

    def encode_act(self, act):
        act = self.action_encoder(act)  # (b, num_frames, action_emb_dim)
        return act

    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        T = z.shape[1]
        # reshape to a batch of windows of inputs
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs, z_act = self.separate_emb(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
        visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            "proprio": z_obs["proprio"],
        }
        return obs, diff

    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                z[..., -(self.proprio_dim + self.action_dim):-self.action_dim], \
                z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def forward(self, obs, act, epoch: int = 0):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
                epoch: used for bisimulation loss
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        z = self.encode(obs, act)
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
        z_tgt = z[:, self.num_pred:, :, :]  # (b, num_hist, num_patches, dim)
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred:, ...]  # (b, num_hist, 3, img_size, img_size)

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None and not self.has_bisim:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                decoder_loss_pred = (
                        recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            # compute loss for visual, proprio dims (i.e. exclude action dims)
            if self.concat_dim == 0:
                z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
            elif self.concat_dim == 1:
                z_visual_loss = self.emb_criterion(
                    z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                    z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                )
                z_proprio_loss = self.emb_criterion(
                    z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim],
                    z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                )
                z_loss = self.emb_criterion(
                    z_pred[:, :, :, :-self.action_dim],
                    z_tgt[:, :, :, :-self.action_dim].detach()
                )

            loss = loss + z_loss
            loss_components["z_loss"] = z_loss
            loss_components["z_visual_loss"] = z_visual_loss
            loss_components["z_proprio_loss"] = z_proprio_loss

            # add bisimulation loss
            if self.has_bisim:
                z_obs_src, z_act_src = self.separate_emb(z_src)
                z_obs_pred, _ = self.separate_emb(z_pred)
                action_emb = self.encode_act(act[:, : self.num_hist])

                bisim_loss, z_dist, r_dist, transition_dist, var_loss, cov_reg = self.calc_bisim_loss(
                    z_obs_src["visual"],
                    z_obs_pred["visual"],
                    action_emb,
                    epoch=epoch
                )
                bisim_loss = bisim_loss.mean()
                loss = loss + self.bisim_coef * bisim_loss

                loss_components["bisim_loss"] = bisim_loss
                loss_components["bisim_z_dist"] = z_dist
                loss_components["bisim_r_dist"] = r_dist
                loss_components["bisim_transition_dist"] = transition_dist
                loss_components["bisim_var_loss"] = var_loss
                loss_components["bisim_cov_reg"] = cov_reg

        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None and not self.has_bisim:
            obs_reconstructed, diff_reconstructed = self.decode(
                z.detach()
            )  # recon loss should only affect decoder
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            decoder_loss_reconstructed = (
                    recon_loss_reconstructed
                    + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
        else:
            visual_reconstructed = None
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z

    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:]
        z = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist:])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t: t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist:])
        z_new = z_pred[:, -1:, ...]  # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses, z_acts = self.separate_emb(z)

        return z_obses, z
