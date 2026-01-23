import os
import sys
import torch
import torch.nn as nn


class SimDINOv2Encoder(nn.Module):
    """
    SimDINOv2 Encoder wrapper that matches the DinoV2Encoder interface.
    
    Key differences from DINOv2:
    - Uses ViT-B/16 (patch_size=16) instead of ViT-S/14 (patch_size=14)
    - Embedding dimension is 768 instead of 384
    - With 224x224 input: (224/16)^2 = 196 patches (same as DINOv2 after cropping)
    - No image resizing needed (uses 224x224 directly)
    """
    
    def __init__(self, name, checkpoint_path, feature_key):
        super().__init__()
        self.name = name
        self.feature_key = feature_key
        
        _simdino_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "SimDINO")
        if _simdino_path not in sys.path:
            sys.path.insert(0, _simdino_path)
        
        from simdinov2.models.vision_transformer import vit_base
        
        num_register_tokens = 4 if "reg4" in name else 0
        
        self.base_model = vit_base(
            patch_size=16,
            num_register_tokens=num_register_tokens,
            img_size=224,
            block_chunks=0,
        )
        
        model_keys = list(self.base_model.state_dict().keys())
        chunked_format = any('blocks.0.' in k and '.' in k.split('blocks.0.')[1].split('.')[0] for k in model_keys if 'blocks.0.' in k)
        if chunked_format:
            print(f"  WARNING: Model has chunked blocks")
            print(f"  Sample model keys: {model_keys[:5]}")
        
        self.emb_dim = self.base_model.embed_dim  # 768 for vit_base
        self.patch_size = self.base_model.patch_size  # 16
        
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")
        
        self._load_checkpoint(checkpoint_path)
        
        print(f"Initialized SimDINOv2Encoder: {name}")
        print(f"  - embed_dim: {self.emb_dim}")
        print(f"  - patch_size: {self.patch_size}")
        print(f"  - num_register_tokens: {num_register_tokens}")
        print(f"  - feature_key: {feature_key}")
    
    def _load_checkpoint(self, checkpoint_path):
        if not os.path.isabs(checkpoint_path):
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint_path = os.path.join(base_path, checkpoint_path)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading SimDINOv2 checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        state_dict = checkpoint
        
        for key in ["teacher", "model", "state_dict"]:
            if key in state_dict:
                print(f"  Found checkpoint key: {key}")
                state_dict = state_dict[key]
                break
        
        prefixes_to_remove = ["_orig_mod.", "backbone.", "module."]
        for prefix in prefixes_to_remove:
            state_dict = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dino_head')}
        
        checkpoint_sample = [k for k in state_dict.keys() if 'blocks.' in k][:5]
        
        import re
        has_chunked = any(re.match(r'blocks\.\d+\.\d+\.', k) for k in checkpoint_sample)
        if has_chunked:
            from simdinov2.utils.utils import revert_block_chunk_weight
            state_dict = revert_block_chunk_weight(state_dict)
        
        msg = self.base_model.load_state_dict(state_dict, strict=False)
        
        unexpected_filtered = [k for k in msg.unexpected_keys if not ('ls1' in k or 'ls2' in k)]
        missing_filtered = [k for k in msg.missing_keys if not k.startswith('dino_head')]
        
        if missing_filtered:
            print(f"  Warning: {len(missing_filtered)} missing keys")
            if len(missing_filtered) <= 5:
                print(f"    Missing: {missing_filtered}")
        if unexpected_filtered:
            print(f"  Warning: {len(unexpected_filtered)} unexpected keys")
            if len(unexpected_filtered) <= 5:
                print(f"    Unexpected: {unexpected_filtered[:5]}")
        
        loaded_blocks = set()
        for k in state_dict.keys():
            if k.startswith('blocks.') and '.norm1.weight' in k:
                block_num = k.split('blocks.')[1].split('.')[0]
                loaded_blocks.add(block_num)
        print(f"  Successfully loaded weights for {len(loaded_blocks)} transformer blocks")
    
    def forward(self, x):
        """
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
               Expected: (B, 3, 224, 224)
        
        Returns:
            Embeddings of shape:
            - (B, num_patches, emb_dim) for x_norm_patchtokens
            - (B, 1, emb_dim) for x_norm_clstoken (with dummy patch dim)
        """
        features = self.base_model.forward_features(x)
        emb = features[self.feature_key]
        
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1)
        
        return emb
