import logging
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the local DINOv3 repo importable as the top-level `dinov3` package.
_this_dir = os.path.dirname(os.path.abspath(__file__))
# /home/.../SpineAgent/report_generation/llava/model/multimodal_encoder
# -> /home/.../SpineAgent
_project_root = os.path.abspath(os.path.join(_this_dir, "../../../.."))
_pretrain_root = os.path.join(_project_root, "pretraining")
if os.path.isdir(_pretrain_root) and _pretrain_root not in sys.path:
    sys.path.append(_pretrain_root)

from dinov3.configs import DinoV3SetupArgs, get_cfg_from_args
from dinov3.models import build_model_for_eval as build_dinov3_model_for_eval

logging.getLogger("dual_dinov3_encoder").setLevel(logging.INFO)


class DualDinoVisionTower(nn.Module):
    def __init__(
        self,
        vision_tower_t1: str,
        vision_tower_t2: str,
        args,
        delay_load: bool = False,
    ):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_t1_name = vision_tower_t1
        self.vision_tower_t2_name = vision_tower_t2
        
        # Store args for later use
        self.args = args
        
        # Initialize model paths & config files (DINOv3 checkpoints)
        self.model_path1 = args.model_path_t1
        self.model_path2 = args.model_path_t2
        self.config_file_t1 = getattr(args, "config_file_t1", None)
        self.config_file_t2 = getattr(args, "config_file_t2", None)
        
        # Initialize configs
        self.cfg_t1 = None
        self.cfg_t2 = None
        
        # Vision configuration
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.fusion_method = getattr(args, 'dual_vision_fusion_method', 'concat')

        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            # Eagerly load full DINOv3 backbones.
            self.load_model()
        
        self._dtype = torch.float32 
        self.image_processor = None

    def to(self, *args, **kwargs):
        """
        Overrides the nn.Module.to() method to correctly handle the two vision towers
        and update the internal dtype.
        """
        # Call the original to() method to move all parameters and buffers
        super().to(*args, **kwargs)

        # Update the internal dtype if it's provided
        if 'dtype' in kwargs:
            self._dtype = kwargs['dtype']

        # Ensure submodules are correctly moved if they have special logic (unlikely for DINOv2 but good practice)
        if self.is_loaded:
            self.vision_tower_t1.to(*args, **kwargs)
            self.vision_tower_t2.to(*args, **kwargs)
        
        return self

    def _build_cfg(self, config_file: Optional[str]):
        """
        Build a DINOv3 config object from a YAML config file.
        This mirrors the official DINOv3 setup but avoids requiring
        distributed initialization.
        """
        if config_file is None:
            return None

        setup_args = DinoV3SetupArgs(
            config_file=config_file,
            pretrained_weights=None,
            shard_unsharded_model=False,
            output_dir="",
            opts=[],
        )
        # Merge default ssl_default_config with the provided config.
        cfg = get_cfg_from_args(setup_args, strict=False)
        return cfg

    def _resolve_pretrained_path(self, cfg, explicit_path: Optional[str]) -> Optional[str]:
        """
        Decide which checkpoint path to use for a given tower.

        Priority:
        1) Explicit path from training args (model_path_t1 / model_path_t2)
        2) student.resume_from_teacher_chkpt from DINOv3 config
        3) gram.ckpt from DINOv3 config
        """
        if explicit_path:
            return explicit_path
        # Try typical DINOv3 locations in the config.
        path = None
        if hasattr(cfg, "student") and getattr(cfg.student, "resume_from_teacher_chkpt", ""):
            path = cfg.student.resume_from_teacher_chkpt
        if (not path) and hasattr(cfg, "gram") and getattr(cfg.gram, "ckpt", ""):
            path = cfg.gram.ckpt
        return path or None

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'Dual DINO models are already loaded, `load_model` called again, skipping.')
            return

        # Build configs on first load if needed.
        if self.cfg_t1 is None:
            if self.config_file_t1 is None:
                raise ValueError("config_file_t1 must be provided to load DINOv3 T1 backbone.")
            self.cfg_t1 = self._build_cfg(self.config_file_t1)
        if self.cfg_t2 is None:
            if self.config_file_t2 is None:
                raise ValueError("config_file_t2 must be provided to load DINOv3 T2 backbone.")
            self.cfg_t2 = self._build_cfg(self.config_file_t2)

        # Resolve checkpoints from args / configs.
        weights_t1 = self._resolve_pretrained_path(self.cfg_t1, self.model_path1)
        weights_t2 = self._resolve_pretrained_path(self.cfg_t2, self.model_path2)

        # Build DINOv3 teacher backbones for t1/t2.
        self.vision_tower_t1 = build_dinov3_model_for_eval(self.cfg_t1, weights_t1)
        self.vision_tower_t2 = build_dinov3_model_for_eval(self.cfg_t2, weights_t2)
        # Register models as submodules so they appear in model architecture
        self.add_module('vision_tower_t1', self.vision_tower_t1)
        self.add_module('vision_tower_t2', self.vision_tower_t2)
        
        # Freeze vision towers (typically we don't fine-tune the vision encoders)
        self.vision_tower_t1.requires_grad_(False)
        self.vision_tower_t2.requires_grad_(False)
        
        logging.info(f'Loaded Dual DINOv3 models: {self.vision_tower_t1_name} and {self.vision_tower_t2_name}')
        logging.info(f'Fusion method: {self.fusion_method}, Feature selection: {self.select_feature} from layer {self.select_layer}')

        
        self.is_loaded = True
        if device_map is not None:
            self.to(device_map)
            logging.info(f'Moved Dual DINOv2 models to device map: {device_map}')


    def feature_select(self, image_forward_outs):
        if self.select_layer == -1:
            # Use last hidden state
            image_features = image_forward_outs.last_hidden_state
        else:
            raise NotImplementedError("Currently only select_layer=-1 is supported.")
        
        if self.select_feature == 'patch':
            # Remove CLS token, keep only patch tokens (include register tokens)
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls':
            # Keep only CLS token
            image_features = image_features[:, 0:1]
        elif self.select_feature == 'cls_patch':
            # Keep all tokens (CLS + patch)
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        
        return image_features
    
    def fuse_features(self, features_t1: torch.Tensor, features_t2: torch.Tensor) -> torch.Tensor:
        """Fuse features from two models"""
        if self.fusion_method == 'concat':
            # Concatenate features along seq
            return torch.cat([features_t1, features_t2], dim=1)
        else:
            raise ValueError(f'Unknown fusion method: {self.fusion_method}')
    
    @ torch.no_grad()
    def forward(self, images_t1, images_t2):
        if type(images_t1) is list and type(images_t2) is list:
            features_t1 = []
            for image in images_t1:
                # Process single image through both models
                image_tensor = image.to(device=self.device, dtype=self.dtype)
                image_forward_t1 = self.vision_tower_t1(image_tensor) # [slices, 1024]
                # print("image forward t1 shape: ", image_forward_t1.shape)
                features_t1.append(image_forward_t1)
            features_t2 = []
            for image in images_t2:
                image_tensor = image.to(device=self.device, dtype=self.dtype)
                image_forward_t2 = self.vision_tower_t2(image_tensor)
                features_t2.append(image_forward_t2)
            image_features = [torch.cat([f1, f2], dim=0) for f1, f2 in zip(features_t1, features_t2)]
        elif type(images_t1) is torch.Tensor and type(images_t2) is torch.Tensor:
            assert images_t1.dim() == 4, f'Expected images_t1 to be a 4D tensor, got {images_t1.dim()}D'
            assert images_t2.dim() == 4, f'Expected images_t2 to be a 4D tensor, got {images_t2.dim()}D'
            # Process batch of
            images_t1 = images_t1.to(device=self.device, dtype=self.dtype)
            images_t2 = images_t2.to(device=self.device, dtype=self.dtype)
            image_forward_t1 = self.vision_tower_t1(images_t1)
            image_forward_t2 = self.vision_tower_t2(images_t2)
            image_features = [torch.cat([image_forward_t1, image_forward_t2], dim=0)]
        else:
            print("--------------------------------")
            print("invalid type of images_t1 or images_t2")
            if type(images_t1) is list:
                print(len(images_t1))
            if type(images_t2) is list:
                print(len(images_t2))
            if type(images_t1) is torch.Tensor:
                print(images_t1.shape)
            if type(images_t2) is torch.Tensor:
                print(images_t2.shape)
            print("--------------------------------")
            raise ValueError(f"Invalid type of images_t1: {type(images_t1)}, images_t2: {type(images_t2)}")
        return image_features
    
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def device(self):
        if self.is_loaded:
            return next(self.parameters()).device
        else:
            return torch.device('cpu')
    
    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower_t1.config
        else:
            return self.cfg_t1
    
    @property
    def hidden_size(self):
        """Return the hidden size after fusion"""
        if self.config is None:
            return 1024  # Default fallback value
        base_hidden_size = self.config.hidden_size
        return base_hidden_size
    
    @property
    def num_patches_per_side(self):
        # DINOv2 uses 14x14 patches for 224x224 images by default
        if self.config is None:
            return 14  # Default fallback value
        return self.config.image_size // self.config.patch_size
    
    @property
    def num_patches(self):
        if self.config is None:
            return 196  # Default fallback value (14*14)
        return (self.config.image_size // self.config.patch_size) ** 2
    
    def __repr__(self):
        """Custom representation showing model architecture even when delay_load=True"""
        if self.is_loaded:
            # Use default repr when models are loaded
            return super().__repr__()
        else:
            # Custom repr for unloaded state
            fusion_info = f"fusion_method={self.fusion_method}"
            return (f"DualDinoVisionTower(\n"
                   f"  (vision_tower_t1): {self.vision_tower_t1_name}\n"
                   f"  (vision_tower_t2): {self.vision_tower_t2_name}\n"
                   f"  (fusion_layer): {fusion_info}\n"
                   f"  (feature_select): layer={self.select_layer}, feature={self.select_feature}\n"
                   f"  (status): delay_loaded, models not yet instantiated\n"
                   f")"
                   f"hidden_size after fusion: {self.hidden_size}")