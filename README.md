# PPS Self-Training

**Does Proxy Policy Steering stay reliable when a robot practices on its own?**

[PPS paper](https://arxiv.org/abs/2609.09148) · [Official PPS code](https://github.com/TritiumR/pps) · [Lab log](LOG.md)

```mermaid
flowchart LR
    A[Self-Training Round] --> B[New States Visited]
    B --> C[Fixed Reference]
    B --> D[Refreshed Reference]
    C --> E[Measure Drift + Recovery]
    D --> E
    E --> F[Does Refresh Preserve Recovery?]
```

## Overview

[Proxy Policy Steering](https://arxiv.org/abs/2609.09148) (PPS, Ning, Wang, Ma and Fang, CoRL 2026) adapts a large frozen robot policy to a new task without touching its weights. Two small proxy networks run alongside it.

- A **reference proxy** is distilled to copy the big model (π0.5).
- A **task proxy** starts from the reference and is fine-tuned on about 50 human demos of the new task.

At every denoising step the big model's velocity is nudged by the difference between the two.

```
v = v_base + γ (v_task − v_ref)
```

The difference carries only what the demos added, so the robot keeps the big model's general skills (like recovering from a dropped object) while picking up the new task.

This repo reproduces PPS and then asks what happens when the robot keeps improving from its own practice. Each round, PPS runs the task, its successful runs are added to the task proxy's training data, and the task proxy is retrained. That sends the robot into states the reference proxy never saw.

**The question.** On those new states, does the reference proxy stop being a good copy of the big model? If it drifts, the correction it feeds into the big model is wrong, and recovery skills could quietly break. Does re-copying the reference on the new states fix that, and can it be done cheaply by refreshing only where drift is likely?

## Results so far

LIBERO-90 task 59, π0.5 base, PPS at γ = 0.4, paper-style proxies trained on all 50 demos, RTX PRO 4500. These numbers are being rerun on the RTX 4090 and will be replaced.

| Test | π0.5 alone | PPS | Difference (95% CI) |
| --- | --- | --- | --- |
| Normal, 200 paired trials | 24.5% | 64.5% | +40.0 pts (+29.5 to +50.5) |
| Mid-carry drop, 200 paired branches | 70.0% | 65.5% | −4.5 pts (−12.0 to +2.0) |

PPS clearly improves the task. On the drop test it is not significantly worse than π0.5, but it shows no advantage either. The comparison against methods expected to lose recovery (task proxy alone, LoRA) will show whether PPS actually preserves it. The full summary, `runs/eval_t59/SUMMARY.md`, is on the `stage3-eval-t59` branch.

## Experiments

**Self-training conditions**, about 10 rounds each, every method given the same number of robot runs.

| # | Condition |
| --- | --- |
| 1 | PPS after reproduction only |
| 2 | Task proxy trained longer on the human demos, matched compute |
| 3 | Self-training, reference never refreshed |
| 4 | Self-training, reference refreshed every round |
| 5 | DSRL |
| 6 | Self-training, drift-aware selective refresh |
| 7 | Self-training, drift-aware gated steering (lower γ on flagged states) |
| 8 | UF-OPS |

**Measured every round.** Success rate, recovery after a mid-carry drop, action diversity, the reference proxy's gap to π0.5 on that round's new states, and π0.5 calls spent on refreshing.

**Baselines.** π0.5 alone, task proxy alone, LoRA fine-tuning, from-scratch specialist, residual policy, DSRL, UF-OPS.

**Where.** LIBERO (4 tasks) and RoboCasa (6 tasks) in simulation, with π0.5 on both and π0 on LIBERO. A real low-cost SO-101 arm on a bottle-to-bowl task, with two π0.5 bases fine-tuned on about 100 and about 400 demos, to test PPS where the base had to learn the robot from a few hundred demos.

## Repository layout

```
pps_real/        sampler, policy wrapper, LIBERO observation handling, perturbations
scripts/
  m0/            milestone 0, π0.5 on held-out LIBERO-90 tasks
  stage2/        sampler check, distillation data, perturbation tests on task 59
  toy/           one script per step of the PushT toy
toy/             PushT toy version of PPS
docs/            stage notes (stage2.md, toy.md)
results/         small result files, plots, and archive/ for superseded runs
LOG.md           lab log, what was planned and what actually happened
data/            local only, ignored by git
checkpoints/     local only, ignored by git
```

Not on main yet: `scripts/stage3` (task 59 evaluation, drop test, summaries) lives on the `stage3-eval-t59` branch. Proxy training scripts are not on main either.

## Setup

Tested on Ubuntu 24.04 with an NVIDIA GPU. Evaluation is being standardized on an RTX 4090. Milestone 0 and stage 2 ran on an RTX PRO 6000 and the first stage 3 run on an RTX PRO 4500.

1. **openpi.** Clone [openpi](https://github.com/Physical-Intelligence/openpi) at commit `215abfb` and set up its environment (Python 3.11) and its LIBERO example environment (Python 3.8) with `uv`. On Blackwell GPUs (sm_120) openpi's pinned torch 2.7.1 has no kernels, so add the PyTorch cu128 index as a uv source for torch and torchvision and re-lock. The local openpi changes are in `results/m0/openpi_local.patch`.
2. **Environment variables.** Source `env.sh`. It is not in this repo. It lives at `/workspace/env.sh` on the pod and sets cache paths, `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` and `LIBERO_CONFIG_PATH=/workspace/.libero`. Create `/workspace/.libero/config.yaml` first so LIBERO does not ask for a config path on first import.
3. **Rendering on a fresh machine.** Install the EGL loader if MuJoCo rendering fails.

```
apt-get install -y libegl1 libgl1 libglvnd0
```

4. **Checkpoints.** π0.5 LIBERO (`pi05_libero`), converted to PyTorch and placed at `/workspace/checkpoints/pi05_libero_pytorch`. Copy the assets folder (norm stats) into the converted checkpoint by hand, because the converter does not.
5. **Toy only.** Python is pinned to 3.12 and every dependency is pinned in `pyproject.toml` and `uv.lock`. Run `uv sync`.

RunPod's nginx holds port 8001, so servers use 8000 and 8002 to 8005.

## Usage

Each script's docstring gives its exact command. The main ones follow.

**Check the steering sampler against openpi's stock sampler.**

```
/workspace/openpi/.venv/bin/python scripts/stage2/check_sampler.py --mode eager
```

**Evaluate π0.5 on LIBERO-90 tasks.** Start a policy server, then run episodes against it.

```
/workspace/openpi/.venv/bin/python scripts/m0/serve.py --checkpoint /workspace/checkpoints/pi05_libero_pytorch --port 8000
python scripts/m0/eval_libero.py --suite libero_90 --task-ids 59 --episodes 30 --port 8000 --out results/raw/x.jsonl
```

`eval_libero.py` runs in openpi's LIBERO venv (`/workspace/openpi/examples/libero/.venv/bin/python`).

**Run the task 59 perturbation tests on the base.**

```
PYTHONPATH=. /workspace/openpi/.venv/bin/python scripts/stage2/serve.py --port 8002
PYTHONPATH=.:/workspace/openpi/third_party/libero /workspace/openpi/examples/libero/.venv/bin/python \
    scripts/stage2/eval_perturb.py --perturbation teleport --episodes 10 --port 8002 \
    --out results/stage2/perturb/teleport.jsonl
```

**Evaluate π0.5 and PPS on task 59** (on the `stage3-eval-t59` branch). Start a policy server, then run trials against it.

```
PYTHONPATH=. /workspace/openpi/.venv/bin/python scripts/stage3/serve_eval.py --port 8010
PYTHONPATH=.:/workspace/openpi/third_party/libero /workspace/openpi/examples/libero/.venv/bin/python \
    scripts/stage3/run_t59.py paired --port 8010 --method pps --trial-list 0 1 2
```

**Run the PushT toy end to end**, with three proxy seeds. About 25 minutes on an M5 laptop.

```
uv run python scripts/toy/train_base.py
uv run python scripts/toy/make_task.py
for s in 0 1 2; do uv run python scripts/toy/distill_ref.py --seed $s && uv run python scripts/toy/train_task.py --seed $s; done
uv run python scripts/toy/eval.py --seeds 0 1 2
```

## Reproducibility notes

- Every comparison uses one GPU type. bf16 kernels differ across GPUs, so the same trial can succeed on one and fail on another.
- The sampler runs eager (no `torch.compile`), batch 1.
- Each trial fixes its LIBERO init state, environment seed, and the noise for every policy call, so two methods see identical conditions until their actions diverge.

## Acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) (Physical Intelligence), [LIBERO](https://libero-project.github.io/), and the [official PPS code](https://github.com/TritiumR/pps). Proxies use DINOv3 ViT-S/16 weights.

## References

**Method this repo builds on**

1. Ning, Wang, Ma, Fang. *Proxy Policy Steering.* 2026. CoRL 2026 per the [official code page](https://github.com/TritiumR/pps), the arXiv page lists no venue. [arXiv 2609.09148](https://arxiv.org/abs/2609.09148) · [code](https://github.com/TritiumR/pps)

**Base models**

2. Black, Brown, Driess et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* RSS 2025 (arXiv 2024). [arXiv 2410.24164](https://arxiv.org/abs/2410.24164)
3. Physical Intelligence, Black, Brown, Darpinian et al. *π0.5: a Vision-Language-Action Model with Open-World Generalization.* 2025. [arXiv 2504.16054](https://arxiv.org/abs/2504.16054) · [openpi](https://github.com/Physical-Intelligence/openpi)
4. Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023. [arXiv 2210.02747](https://arxiv.org/abs/2210.02747)
5. Siméoni, Vo, Seitzer et al. *DINOv3.* 2025. [arXiv 2508.10104](https://arxiv.org/abs/2508.10104)

**Benchmarks**

6. Liu, Zhu, Gao, Feng, Liu, Zhu, Stone. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.* NeurIPS 2023 Datasets and Benchmarks. [arXiv 2306.03310](https://arxiv.org/abs/2306.03310)
7. Nasiriany, Maddukuri, Zhang, Parikh, Lo, Joshi, Mandlekar, Zhu. *RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots.* RSS 2024. [arXiv 2406.02523](https://arxiv.org/abs/2406.02523)

**Baselines**

8. Wagenmaker, Nakamoto, Zhang, Park, Yagoub, Nagabandi, Gupta, Levine. *Steering Your Diffusion Policy with Latent Space Reinforcement Learning (DSRL).* CoRL 2025. [arXiv 2506.15799](https://arxiv.org/abs/2506.15799) · [code](https://github.com/nakamotoo/dsrl_pi0)
9. Attarian, Vyse, Gerigk, Opryshko, Ruan, Almasri, Singh, Du, Gilitschenski, Voelcker. *Update-Free On-Policy Steering via Verifiers (UF-OPS).* 2026. [arXiv 2603.10282](https://arxiv.org/abs/2603.10282)
10. Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [arXiv 2106.09685](https://arxiv.org/abs/2106.09685)

**Related steering and composition work**

11. Im, Kolobov, Fu, Lee. *Latent Policy Steering through One-Step Flow Policies (LPS).* RSS 2026. [arXiv 2603.05296](https://arxiv.org/abs/2603.05296)
12. Tang, Chen, Wagenmaker, Finn, Levine. *Improving Robotic Generalist Policies via Flow Reversal Steering.* 2026. [arXiv 2606.13675](https://arxiv.org/abs/2606.13675)
13. Kao, Chan, Wang. *Flow Control: Steering Vision-Language-Action Models with Simple Real-Time Inputs.* CoRL 2026. [arXiv 2606.10180](https://arxiv.org/abs/2606.10180)
14. Liu, Li, Du, Torralba, Tenenbaum. *Compositional Visual Generation with Composable Diffusion Models.* ECCV 2022. [arXiv 2206.01714](https://arxiv.org/abs/2206.01714)
15. Wang, Zhao, Du, Adelson, Tedrake. *PoCo: Policy Composition from and for Heterogeneous Robot Learning.* RSS 2024. [arXiv 2402.02511](https://arxiv.org/abs/2402.02511)

**Self-improvement and distillation work**

16. Lin, Shakkottai, Kalathil, Kumar. *ReGuide: From Test-Time Guidance to Self-Improving Diffusion Policies.* 2026. [arXiv 2606.28939](https://arxiv.org/abs/2606.28939)
17. Giridhar, Khandelwal, Collins, Georgiev, Garg. *Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning.* 2026. [arXiv 2608.21204](https://arxiv.org/abs/2608.21204)
18. Ross, Gordon, Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger).* AISTATS 2011. [arXiv 1011.0686](https://arxiv.org/abs/1011.0686)
19. Agarwal, Vieillard, Zhou, Stanczyk, Ramos, Geist, Bachem. *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.* ICLR 2024. [arXiv 2306.13649](https://arxiv.org/abs/2306.13649)
20. Xiao, Lin, Peng, Xue, He, Xie, Hu, Wu, Luo, Fan, Shi, Zhu. *Self-Improving Vision-Language-Action Models with Data Generation via Residual RL (Probe, Learn, Distill).* 2025. [arXiv 2511.00091](https://arxiv.org/abs/2511.00091)
21. Oh, Guo, Singh, Lee. *Self-Imitation Learning.* ICML 2018. [arXiv 1806.05635](https://arxiv.org/abs/1806.05635)
22. Nicolicioiu, Pezeshki, Courville. *On-Policy Self-Distillation with Sampled Demonstrations Reduces Output Diversity.* 2026. [arXiv 2606.26091](https://arxiv.org/abs/2606.26091)
