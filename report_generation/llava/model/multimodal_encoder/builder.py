import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .dual_dinov3_encoder import DualDinoVisionTower 


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    # Check if dual DINOv3 setup is requested
    vision_tower_t1 = getattr(vision_tower_cfg, 'vision_tower_t1', None)
    vision_tower_t2 = getattr(vision_tower_cfg, 'vision_tower_t2', None)

    if vision_tower_t1 is not None and vision_tower_t2 is not None:
        # Use dual DINOv3 vision tower
        return DualDinoVisionTower(
            vision_tower_t1=vision_tower_t1, vision_tower_t2=vision_tower_t2,
            args=vision_tower_cfg, **kwargs
        )
    # Original logic for single vision tower
    is_absolute_path_exists = os.path.exists(vision_tower) if vision_tower else False
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if vision_tower and (is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower):
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
