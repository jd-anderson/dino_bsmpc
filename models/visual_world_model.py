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
            bisim_model=None,  # New parameter for bisimulation model
            bisim_latent_dim=64,  # New parameter for bisimulation latent dimension
            bisim_hidden_dim=256,  # New parameter for bisimulation hidden dimension
            bisim_coef=1.0,  # New parameter for bisimulation loss coefficient
            train_bisim=True,  # New parameter to control training of bisimulation model
            proprio_dim=0,
            action_dim=0,
            concat_dim=0,
            num_action_repeat=7,
            num_proprio_repeat=7,
            train_encoder=True,
            train_predictor=False,
            train_decoder=True,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None

        # Initialize bisimulation model if provided or create a new one
        self.has_bisim = bisim_model is not None
        if self.has_bisim:
            self.bisim_model = bisim_model
            self.bisim_latent_dim = bisim_model.latent_dim
            self.train_bisim = train_bisim
            self.bisim_coef = bisim_coef
        else:
            self.bisim_model = None
            self.train_bisim = False
            self.bisim_coef = 0.0

        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat
        self.action_dim = action_dim * num_action_repeat
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim)  # Not used

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")
        if self.has_bisim:
            print(f"bisim_model: {self.bisim_model}")
            print(f"bisim_latent_dim: {self.bisim_latent_dim}")
            print(f"train_bisim: {self.train_bisim}")
            print(f"bisim_coef: {self.bisim_coef}")

        self.concat_dim = concat_dim  # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()

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

    def encode(self, obs, act):
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
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

        # Debug prints for dimensions
        # print(f"DEBUG - DinoV2 output dimensions: {visual_embs.shape}")
        if hasattr(visual_embs, 'flatten'):
            flattened = visual_embs.flatten(2)
            # print(f"DEBUG - DinoV2 flattened dimensions: {flattened.shape}")

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
            "proprio": z_obs["proprio"],  # Note: no decoder for proprio for now!
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

    def encode_bisim(self, z_dino):
        """
        Maps DinoV2 embeddings to bisimulation embeddings
        input: z_dino (dict): {"visual": (b, t, p, d), "proprio": (b, t, d)}
        output: z_bisim: (b, t, bisim_dim)
        """
        if not self.has_bisim:
            return None

        # Add focused log to confirm encoder is being called
        # print(f"BISIM ENCODER: Called encode_bisim with visual shape {z_dino['visual'].shape}")

        # Use only visual embeddings for bisimulation
        z_bisim = self.bisim_model.encode(z_dino["visual"])

        # Log output dimensions
        # print(f"BISIM ENCODER: Output bisimulation embeddings shape: {z_bisim.shape}")

        return z_bisim

    def predict_next_bisim(self, z_bisim, action_emb):
        """
        Predicts next bisimulation state
        input: z_bisim: (b, t, bisim_dim)
               action_emb: (b, t, action_emb_dim)
        output: next_z_bisim: (b, t, bisim_dim)
        """
        if not self.has_bisim:
            return None

        b, t, d = z_bisim.shape
        # print(f"BISIM PREDICTION: Input bisim state shape: {z_bisim.shape}, dim={d}")
        z_bisim_flat = z_bisim.reshape(b * t, d)
        action_emb_flat = action_emb.reshape(b * t, -1)

        next_z_bisim = self.bisim_model.next(z_bisim_flat, action_emb_flat)
        next_z_bisim = next_z_bisim.reshape(b, t, d)
        # print(f"BISIM PREDICTION: Output next bisim state shape: {next_z_bisim.shape}, dim={d}")

        return next_z_bisim

    def calc_bisim_loss(self, z_bisim, action_emb, reward=None, discount=0.99):
        """
        Calculate bisimulation loss
        input: z_bisim: (b, t, bisim_dim)
               action_emb: (b, t, action_emb_dim)
               reward: (b, t, 1) or None (will be predicted)
        output: bisim_loss: (b, t)
        """
        if not self.has_bisim:
            return torch.tensor(0.0, device=z_bisim.device)

        b, t, d = z_bisim.shape
        batch_size = b

        # Debug prints for dimensions
        # print(f"DEBUG - calc_bisim_loss input z_bisim dimensions: {z_bisim.shape}")
        # print(f"DEBUG - calc_bisim_loss input action_emb dimensions: {action_emb.shape}")

        # Permute batch for comparison
        perm = torch.randperm(batch_size, device=z_bisim.device)
        z_bisim2 = z_bisim[:, :, :].clone()[perm]

        # Predict rewards if not provided
        if reward is None:
            z_bisim_flat = z_bisim.reshape(b * t, d)
            action_emb_flat = action_emb.reshape(b * t, -1)

            # Debug prints for flattened dimensions
            # print(f"DEBUG - calc_bisim_loss flattened z_bisim dimensions: {z_bisim_flat.shape}")
            # print(f"DEBUG - calc_bisim_loss flattened action_emb dimensions: {action_emb_flat.shape}")

            reward = self.bisim_model.predict_reward(z_bisim_flat, action_emb_flat)
            reward = reward.reshape(b, t, 1)

        reward2 = reward.clone()[perm]

        # Predict next states
        next_z_bisim = self.predict_next_bisim(z_bisim, action_emb)
        next_z_bisim2 = next_z_bisim.clone()[perm]

        # Calculate bisimulation loss
        bisim_loss = self.bisim_model.calc_bisim_loss(
            z_bisim, z_bisim2, reward, reward2, next_z_bisim, next_z_bisim2, discount
        )

        return bisim_loss
    
    
    # mod 1 function on BSMPC
    def calc_bisim_loss(self, z_bisim, next_z_bisim, action_emb, reward=None, discount=0.99):
        """
        Calculate bisimulation loss
        input: z_bisim: (b, t, bisim_dim)
               action_emb: (b, t, action_emb_dim)
               reward: (b, t, 1) or None (will be predicted)
        output: bisim_loss: (b, t)
        """
        if not self.has_bisim:
            return torch.tensor(0.0, device=z_bisim.device)

        b, t, d = z_bisim.shape
        batch_size = b

        # Debug prints for dimensions
        # print(f"DEBUG - calc_bisim_loss input z_bisim dimensions: {z_bisim.shape}")
        # print(f"DEBUG - calc_bisim_loss input action_emb dimensions: {action_emb.shape}")

        # Permute batch for comparison
        perm = torch.randperm(batch_size, device=z_bisim.device)
        z_bisim2 = z_bisim[:, :, :].clone()[perm]

        # Predict rewards if not provided
        if reward is None:
            z_bisim_flat = z_bisim.reshape(b * t, d)
            action_emb_flat = action_emb.reshape(b * t, -1)

            # Debug prints for flattened dimensions
            # print(f"DEBUG - calc_bisim_loss flattened z_bisim dimensions: {z_bisim_flat.shape}")
            # print(f"DEBUG - calc_bisim_loss flattened action_emb dimensions: {action_emb_flat.shape}")

            reward = self.bisim_model.predict_reward(z_bisim_flat, action_emb_flat)
            reward = reward.reshape(b, t, 1)

        reward2 = reward.clone()[perm]

        next_z_bisim2 = next_z_bisim.clone()[perm]

        # Calculate bisimulation loss
        bisim_loss = self.bisim_model.calc_bisim_loss(
            z_bisim, z_bisim2, reward, reward2, next_z_bisim, next_z_bisim2, discount
        )

        return bisim_loss

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
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

        # Process embeddings with bisimulation if available
        if self.has_bisim:
            z_obs_src, z_act_src = self.separate_emb(z_src)
            z_obs_tgt, z_act_tgt = self.separate_emb(z_tgt)

            # Get bisimulation embeddings
            z_bisim_src = self.encode_bisim(z_obs_src)
            z_bisim_tgt = self.encode_bisim(z_obs_tgt)

            # Get bisimulation next 
            next_z_bisim_src = z_bisim_tgt

            # Calculate bisimulation loss
            bisim_loss = self.calc_bisim_loss(
                z_bisim_src,
                next_z_bisim_src,
                self.encode_act(act[:, : self.num_hist])
            ).mean()

            loss_components["bisim_loss"] = bisim_loss
            loss = loss + self.bisim_coef * bisim_loss

        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                obs_pred, diff_pred = self.decode(
                    z_pred.detach()
                )  # recon loss should only affect decoder
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

            # Compute loss for visual, proprio dims (i.e. exclude action dims)
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
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
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

        # If using bisimulation, also return bisimulation embeddings
        if self.has_bisim:
            z_bisim = self.encode_bisim(z_obses)
            # print(f"ROLLOUT COMPLETE: Returning with bisimulation embeddings of shape {z_bisim.shape}, dim={z_bisim.shape[-1]}")
            return z_obses, z, z_bisim

        # print("ROLLOUT COMPLETE: Returning without bisimulation embeddings")
        return z_obses, z