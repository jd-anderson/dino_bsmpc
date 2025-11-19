import os
import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
from models.bisim import BisimModel  # Import the bisimulation model

from loss_history.loss_csv import append_loss_to_csv

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        self.accelerator = Accelerator(log_with="wandb")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False,  # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.bisim_model = None  # Initialize bisim_model as None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        self.train_bisim = self.cfg.model.get('train_bisim', True)  # Control training of bisimulation model
        self.train_w_std_loss = self.cfg.model.get('train_w_std_loss', True)
        self.train_w_reward_loss = self.cfg.model.get('train_w_reward_loss', True)
        self.accelerate = self.cfg.model.get('accelerate_launch', False)
        log.info(f"Train encoder, predictor, decoder, bisim, train_w_std_loss, train_w_reward_loss:\
            {self.cfg.model.train_encoder},\
            {self.cfg.model.train_predictor},\
            {self.cfg.model.train_decoder},\
            {self.train_bisim},\
            {self.train_w_std_loss},\
            {self.train_w_reward_loss}")

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        # Save bisim_predictor if present (part of the visual world model)
        self._keys_to_save += (
            ["bisim_predictor"]
            if self.train_predictor and self.cfg.get('has_bisim', False)
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += (
            ["bisim_model", "bisim_optimizer"]
            if self.train_bisim and self.cfg.get('has_bisim', False)
            else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.init_optimizers()

        self.epoch_log = OrderedDict()

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            ckpt = {}
            for k in self._keys_to_save:
                if hasattr(self.__dict__[k], "module"):
                    ckpt[k] = self.accelerator.unwrap_model(self.__dict__[k])
                else:
                    ckpt[k] = self.__dict__[k]
            torch.save(ckpt, "checkpoints/model_latest.pth")
            torch.save(ckpt, f"checkpoints/model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def load_ckpt(self, filename="model_latest.pth"):
        ckpt = torch.load(filename)
        for k, v in ckpt.items():
            self.__dict__[k] = v
        not_in_ckpt = set(self._keys_to_save) - set(ckpt.keys())
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)

    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches ** 2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                        + (
                                proprio_emb_dim * self.cfg.num_proprio_repeat
                                + action_emb_dim * self.cfg.num_action_repeat
                        )
                        * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False

        # Initialize bisimulation model
        if self.cfg.get('has_bisim', False):
            if self.bisim_model is None:
                if self.encoder.latent_ndim == 1:  # if feature is 1D
                    input_dim = self.encoder.emb_dim
                else:
                    decoder_scale = 16  # from vqvae
                    num_side_patches = self.cfg.img_size // decoder_scale
                    num_patches = num_side_patches ** 2
                    input_dim = num_patches * self.encoder.emb_dim

                print(
                    f"DEBUG - Initializing BisimModel with: encoder.latent_ndim={self.encoder.latent_ndim}, encoder.emb_dim={self.encoder.emb_dim}")
                print(f"DEBUG - Calculated values: img_size={self.cfg.img_size}, decoder_scale={decoder_scale}")
                print(f"DEBUG - Calculated values: num_side_patches={num_side_patches}, num_patches={num_patches}")
                print(f"DEBUG - Final input_dim for BiSim model: {input_dim}")

                self.bisim_model = BisimModel(
                    input_dim=input_dim,
                    latent_dim=self.cfg.get('bisim_latent_dim', 8),
                    hidden_dim=self.cfg.get('bisim_hidden_dim', 256),
                    action_dim=self.cfg.action_emb_dim,
                    bypass_dinov2=self.cfg.model.get('bypass_dinov2', False),
                    img_size=self.cfg.img_size,
                    num_patches=196,
                    patch_emb_dim=384,
                )
                bypass_mode = self.cfg.model.get('bypass_dinov2', False)
                log.info(f"Initialized bisimulation model with latent dim {self.cfg.get('bisim_latent_dim', 64)}")
                log.info(f"Bypass DinoV2 mode: {bypass_mode}")
                if bypass_mode:
                    log.info("Training obs -> bisim directly, bypassing DinoV2 embeddings")
                else:
                    log.info("Training obs -> DinoV2 -> bisim")

            if not self.train_bisim:
                for param in self.bisim_model.parameters():
                    param.requires_grad = False

        self.encoder, self.predictor, self.decoder, self.bisim_model = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder, self.bisim_model
        )
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            bisim_model=self.bisim_model,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            var_loss_coef=self.cfg.get('var_loss_coef', 1.0),
            PCA1_loss_target=self.cfg.get('PCA1_loss_target', 0.01),
            VC_target=self.cfg.get('VC_target', 1.0),
            num_pcs=self.cfg.get('num_pcs', 10),
            PCAloss_epoch=self.cfg.get('PCAloss_epoch', 10),
            bisim_latent_dim=self.cfg.get('bisim_latent_dim', 8),
            bisim_hidden_dim=self.cfg.get('bisim_hidden_dim', 256),
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
            bisim_coef=self.cfg.get('bisim_coef', 1.0),
            train_bisim=self.train_bisim,
            train_w_std_loss=self.train_w_std_loss,
            train_w_reward_loss=self.train_w_reward_loss,
            accelerate=self.accelerate,
            bypass_dinov2=self.cfg.model.get('bypass_dinov2', False),
            bisim_memory_buffer_size=self.cfg.get('bisim_memory_buffer_size', 0),
            bisim_comparison_size=self.cfg.get('bisim_comparison_size', 20),
        )

        # Prepare / register bisim predictor for training, saving
        if hasattr(self.model, 'bisim_predictor') and self.model.bisim_predictor is not None:
            # move under accelerator
            self.model.bisim_predictor = self.accelerator.prepare(self.model.bisim_predictor)
            # expose for checkpointing
            self.bisim_predictor = self.model.bisim_predictor

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)

        if self.cfg.get('has_bisim', False) and self.train_bisim:
            self.bisim_optimizer = torch.optim.Adam(
                self.bisim_model.parameters(),
                lr=self.cfg.training.get('bisim_lr', 1e-4),
            )
            self.bisim_optimizer = self.accelerator.prepare(self.bisim_optimizer)

        if self.cfg.has_predictor:
            # Include both regular predictor and bisim predictor if they exist
            predictor_params = list(self.predictor.parameters())
            if hasattr(self.model, 'bisim_predictor') and self.model.bisim_predictor is not None:
                predictor_params.extend(list(self.model.bisim_predictor.parameters()))
            
            self.predictor_optimizer = torch.optim.AdamW(
                predictor_params,
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        init_epoch = self.epoch + 1  # epoch starts from 1
        for epoch in range(init_epoch, init_epoch + self.total_epochs):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            self.train()
            self.accelerator.wait_for_everyone()
            self.val()
            self.logs_flash(step=self.epoch)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                # main thread only: launch planning jobs on the saved ckpt
                if (
                        self.cfg.plan_settings.plan_cfg_path is not None
                        and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        for i, data in enumerate(
                tqdm(self.dataloaders["train"], desc=f"Epoch {self.epoch} Train")
        ):
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            self.model.train()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act, self.epoch
            )

            self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()
            if self.cfg.get('has_bisim', False) and self.train_bisim:
                self.bisim_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()
            if self.cfg.get('has_bisim', False) and self.train_bisim:
                self.bisim_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor and z_out is not None:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist:]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                            self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def val(self):
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset, mode="train"
                )
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
                tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act, self.epoch
            )

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor and z_out is not None:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist:]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                            self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def openloop_rollout(
            self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip

            for k in obs.keys():
                obs[k] = obs[k][
                         start:
                         start + horizon * self.cfg.frameskip + 1:
                         self.cfg.frameskip
                         ]
            act = act[start: start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                # Check if we're using bisimulation model
                if self.cfg.get('has_bisim', False):
                    z_obses, z, z_bisim = self.model.rollout(obs_0, actions)
                    z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                    div_loss = self.err_eval_single(z_obs_last, z_g)

                    # Also calculate bisimulation distance
                    z_bisim_last = z_bisim[:, -1:, :]
                    # Handle bypass mode for goal encoding
                    if self.cfg.model.get('bypass_dinov2', False):
                        z_bisim_g = self.model.encode_bisim(obs_g)
                    else:
                        z_bisim_g = self.model.encode_bisim(z_g)
                    bisim_dist = torch.norm(z_bisim_last - z_bisim_g, dim=-1).mean()

                    log_key = f"bisim_dist_rollout{postfix}"
                    if log_key in logs:
                        logs[log_key].append(bisim_dist.item())
                    else:
                        logs[log_key] = [bisim_dist.item()]
                else:
                    z_obses, z = self.model.rollout(obs_0, actions)
                    z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                    div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                Validation loss: {epoch_log['val_loss']:.4f}")
        log.info(f"Train:  Bisim_loss: {epoch_log['train_bisim_loss']:.4f}  \
                Standard_L2_Loss: {epoch_log['train_standard_l2_loss']:.4f}  z_proprio_loss: {epoch_log['train_z_proprio_loss']:.4f} \
                Bisim z_dist: {epoch_log['train_bisim_z_dist']:.4f}  Bisim r_dist: {epoch_log['train_bisim_r_dist']:.4f}  \
                Variance loss: {epoch_log['train_bisim_var_loss']:.4f}  Transition_dist (Covariance): {epoch_log['train_bisim_transition_dist']:.4f} \
                Covariance loss: {epoch_log['train_bisim_cov_reg']:.4f}")
        log.info(f"Validation:  Bisim_loss: {epoch_log['val_bisim_loss']:.4f}  \
                Standard_L2_Loss: {epoch_log['val_standard_l2_loss']:.4f}  z_proprio_loss: {epoch_log['val_z_proprio_loss']:.4f} \
                Bisim z_dist: {epoch_log['val_bisim_z_dist']:.4f}  Bisim r_dist: {epoch_log['val_bisim_r_dist']:.4f}  \
                Variance loss: {epoch_log['val_bisim_var_loss']:.4f}  Transition_dist (Covariance): {epoch_log['val_bisim_transition_dist']:.4f} \
                Covariance loss: {epoch_log['val_bisim_cov_reg']:.4f}")
        
        append_loss_to_csv(epoch_log, "training_loss_log.csv")

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()

    def plot_samples(
            self,
            gt_imgs,
            pred_imgs,
            reconstructed_gt_imgs,
            epoch,
            batch,
            num_samples=2,
            phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
