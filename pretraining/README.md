### SpineAgent Pretraining Overview

This folder contains the self-supervised pretraining pipeline for the spinal MRI foundation model. It is built on top of the official DINOv3 codebase.

- **Phase 1 – DINOv3 pretraining (T1 / T2 encoders)**  
  - Scripts:  
    - `run_dinov3_pretrain_t1.sh` – trains the T1 encoder.  
    - `run_dinov3_pretrain_t2.sh` – trains the T2 encoder.  
  - Config: `dinov3/configs/train/dinov3_vitl16_pretrain.yaml`  

- **Phase 2 – Gram-regularized “anchor” training**  
  - Scripts:  
    - `run_dinov3_spinal_gram_t1.sh` – Gram phase for the T1 encoder.  
    - `run_dinov3_spinal_gram_t2.sh` – Gram phase for the T2 encoder.  
  - Config: `dinov3/configs/train/dinov3_vitl16_gram_anchor.yaml`  

### Custom dataset class

The main pretraining script defines a **custom dataset** tailored to your spine MRI data:

- File: `dinov3/train/train.py`, class `NewDataset`.  

You can adapt this dataset for your own project.




