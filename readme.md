<div align="center">

# 🚗 AD-H: Autonomous Driving with Hierarchical Agents

**Language-guided Autonomous Driving with Hierarchical Agents**

[![arXiv](https://img.shields.io/badge/arXiv-2406.03474-b31b1b.svg)](https://arxiv.org/abs/2406.03474)

</div>

## 🔥 Updates

- **[2024-06]** AD-H is released on [arXiv](https://arxiv.org/abs/2406.03474).
- **[2025-05]** Inference/evaluation code is released.

## 📖 Overview

AD-H is a **hierarchical multi-agent framework** for language-guided autonomous driving that explicitly separates high-level decision-making from low-level vehicle execution:

- **🧠 MLLM Planner (LLaVA-7B-v1.5 / Mipha-3B)** — Interprets natural-language commands and environmental context to generate coherent mid-level driving instructions (e.g., *"Approaching a junction, prepare to follow traffic rules. Slow down and make a slight left turn."*)
- **⚡ Lightweight Controller (OPT-350M)** — Converts mid-level instructions into precise, continuous control signals via waypoints

Instead of using a single end-to-end MLLM to map language directly to actions, AD-H leverages this decomposition to **unleash the reasoning power of MLLMs** while ensuring stable actuation — even the Mipha-3B variant outperforms 7B-scale single-agent baselines (3B+350M vs. 7B).

### ✨ Key Highlights

- 🏆 **Outperforms state-of-the-art** despite using nearly half the parameters
- 🔄 **Strong emergent generalization** — self-corrects in unseen corner cases (e.g., oversteering)
- 📏 **Robust long-horizon instruction following** — coherent planning across extended temporal sequences
- 📊 **1.15M hierarchical annotation pairs** — constructed via a rule-based pipeline from 26 atomic sub-commands across Perception, Speed, Steer, and Brake dimensions

## 🗂️ Method

| Component | Model | Role |
|-----------|-------|------|
| Planner | LLaVA-7B-v1.5 / Mipha-3B | Decomposes high-level instructions → mid-level driving commands |
| Controller | OPT-350M + Vision Encoder (R50 + Q-Former) | Decodes mid-level commands → waypoints → control signals (PID) |

Mid-level commands are composed from **26 atomic sub-commands** across four dimensions:

| Dimension | Examples |
|-----------|----------|
| 🚦 Perception | *"There is a pedestrian ahead"* / *"Traffic light is red"* |
| 🏎️ Speed | *"Maintain current speed"* / *"Accelerate gradually"* |
| 🔀 Steer | *"Make a slight left turn"* / *"Keep steering straight"* |
| 🛑 Brake | *"Apply brakes safely"* |

> These sub-commands combine to produce **170+ distinct mid-level driving commands**.

## 🚀 Validation

To facilitate convenient inference, we quantize the model during inference, allowing both the CARLA simulator and our model to run simultaneously on a **single GPU with 24GB VRAM**.

💡 For full-model validation on larger VRAM, modify `planner_load_8bit` / `planner_load_4bit` in `team_code/adh_agent_config.py`. For multi-GPU setups, set `planner_device`, `controller_device`, and `CARLA_DEVICE` accordingly.

### 🖥️ Hardware & System

- GPU with at least 24GB VRAM
- Ubuntu 22.04 / 20.04

### 📦 Installation

```bash
git clone https://github.com/zhangzaibin/AD-H.git
cd AD-H
conda env create -f requirements.yml   # If it fails, try changing the conda source
conda activate adh
```

### 🎮 Download CARLA 0.9.15 & Additional Maps

Click to download: [CARLA 0.9.15](https://tiny.carla.org/carla-0-9-15-linux) | [AdditionalMaps 0.9.15](https://tiny.carla.org/additional-maps-0-9-15-linux)

Or via command line:

```bash
wget https://carla-releases.b-cdn.net/Linux/CARLA_0.9.15.tar.gz
wget https://carla-releases.b-cdn.net/Linux/AdditionalMaps_0.9.15.tar.gz
mkdir CARLA_0.9.15
tar -xvf CARLA_0.9.15.tar.gz -C CARLA_0.9.15
mv AdditionalMaps_0.9.15.tar.gz CARLA_0.9.15/Import
cd CARLA_0.9.15 && bash ImportAssets.sh && cd ..
```

### 🔻 Download Pre-trained Models

TODO — checkpoints will be released soon.

Expected `./checkpoints` structure:

```
checkpoints
    ├── llava15-ours/
    ├── opt-350m-ours/
    ├── opt-350m-ours.pth
    └── vision_weights/
        └── vision-encoder-r50.pth.tar
```

### ▶️ Run Evaluation

Place weights in `checkpoints/` or modify paths in `team_code/adh_agent_config.py`, then:

```bash
conda activate adh
bash adh_evaluation.sh
```

## 📋 TODO

- [ ] Release pre-trained model checkpoints
- [ ] Release training code & dataset
- [ ] Release full training data (1.15M hierarchical annotation pairs)

## 📄 Citation

```bibtex
@article{zhang2024adh,
  title={AD-H: Language-guided Autonomous Driving with Hierarchical Agents},
  author={Zhang, Zaibin and Fu, Talas and Tang, Shiyu and Zhang, Yuanhang and Wang, Yifan and Wang, Lijun and Lu, Huchuan},
  journal={arXiv preprint arXiv:2406.03474},
  year={2024}
}
```

## 📜 License

This project is released under the [MIT License](LICENSE).
