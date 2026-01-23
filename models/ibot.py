import os
import torch
import torch.nn as nn
import timm


class IBOTEncoder(nn.Module):
    def __init__(self, name: str, checkpoint_path: str, feature_key: str = "x_norm_patchtokens"):
        super().__init__()
        self.name = name
        self.feature_key = feature_key
        
        if feature_key != "x_norm_patchtokens":
            raise ValueError(f"IBOTEncoder only supports 'x_norm_patchtokens', got: {feature_key}")
        
        # ViT-S/16 @ 224
        self.base_model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="",
        )
        self.patch_size = 16
        self.emb_dim = self.base_model.embed_dim  # 384
        self.latent_ndim = 2  # patch tokens: (B, num_patches, emb_dim)
        
        self._load_checkpoint(checkpoint_path)
        
        print(f"Initialized IBOTEncoder: {name}")
        print(f"  - embed_dim: {self.emb_dim}")
        print(f"  - patch_size: {self.patch_size}")
        print(f"  - feature_key: {feature_key}")

    def _load_checkpoint(self, checkpoint_path: str):
        if not os.path.isabs(checkpoint_path):
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_path = os.path.join(base_path, checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading IBOT checkpoint from: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "teacher" in ckpt:
            print(f"  Found checkpoint key: teacher")
            sd = ckpt["teacher"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            print(f"  Found checkpoint key: model")
            sd = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            print(f"  Found checkpoint key: state_dict")
            sd = ckpt["state_dict"]
        elif isinstance(ckpt, dict):
            sd = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

        prefixes_to_remove = ["module.", "backbone.", "_orig_mod."]
        for prefix in prefixes_to_remove:
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

        drop_prefixes = ("head.", "fc.", "classifier.", "proj.", "predictor.", "dino_head.", "ibot_head.")
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}

        msg = self.base_model.load_state_dict(sd, strict=False)

        core_missing = [k for k in msg.missing_keys if k.startswith("blocks.")]
        if len(core_missing) > 0:
            raise RuntimeError(
                f"Backbone load failed: missing {len(core_missing)} core block keys. "
                f"Example missing: {core_missing[:5]}"
            )
        
        if msg.missing_keys:
            non_block_missing = [k for k in msg.missing_keys if not k.startswith("blocks.")]
            if non_block_missing:
                print(f"  Warning: {len(non_block_missing)} missing keys")
        if msg.unexpected_keys:
            print(f"  Warning: {len(msg.unexpected_keys)} unexpected keys")
        
        loaded_blocks = set()
        for k in sd.keys():
            if k.startswith('blocks.') and '.norm1.weight' in k:
                block_num = k.split('blocks.')[1].split('.')[0]
                loaded_blocks.add(block_num)
        print(f"  Successfully loaded weights for {len(loaded_blocks)} transformer blocks")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        toks = self.base_model.forward_features(x)
        toks = toks[:, 1:, :]
        
        return toks
