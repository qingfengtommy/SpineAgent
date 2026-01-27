# SpineAgent

SpineAgent is a multi-agent framework for spine MRI report generation. It is built on a multi-sequence MRI foundation model trained on routine clinical data and designed to integrate evidence across sequences while preserving sequence-specific diagnostic information. The system decomposes report generation into clinically grounded subtasks handled by specialized agents for diagnosis, pathological slice identification, lesion localization, and retrieval of similar cases.

## Highlights
- Multi-sequence MRI foundation model with separate T1/T2 encoders and continual training via a synthesizer for other sequences.
- Patient-level embeddings that fuse multi-sequence signals for robust downstream tasks.
- State-of-the-art performance on 17 spinal condition prediction tasks and strong cross-manufacturer/cohort generalization.
- Supports pathology localization, image-report retrieval, and scalable report generation.
- End-to-end report generation with explicit agent outputs as tokens in a Medical Report Agent.

## Repository Structure
The repository is organized into five main directories:
- `pretraining/`: DINOv3-based encoder pretraining and continual training scripts.
- `report_generation/`: Medical Report Agent training and inference pipelines.
- `model_alignment/`: Alignment, calibration, and agent-output integration utilities.
- `downstream_tasks/`: Classification, localization, and retrieval tasks.
- `experiments_eval/`: Experiment configs, evaluation scripts, and metrics.

## Install
1. Create a Python environment (e.g., Python 3.10+).
2. Install dependencies:
   - `pip install -r requirements.txt` (if present)
3. Optional: configure GPU/CUDA according to your system.

## Data
This project is trained on a large, de-identified clinical spine MRI cohort. Due to privacy restrictions, the raw data is not included.
- Provide your own MRI data in the expected format.
- See `experiments_eval/` for dataset configuration templates (if available).

## Usage
This repository contains multiple pipelines. Typical entry points include:
- Pretraining encoders in `pretraining/`
- Running downstream tasks in `downstream_tasks/`
- Training or evaluating report generation in `report_generation/`

If the repository includes scripts or configs, follow the README files inside each directory for task-specific instructions.

## Results (Brief)
- Mean 8.14% AUROC improvement across 17 spinal condition prediction tasks over the strongest baseline.
- Effective pathology localization and image-report retrieval.
- Report generation performance approaching expert radiologist assessment (see paper for details).

## Citation
If you use this repository, please cite the SpineAgent paper:
```
@article{spineagent,
  title={SpineAgent: Multi-Agent Spine MRI Report Generation},
  author={Anonymous},
  journal={arXiv},
  year={2026}
}
```