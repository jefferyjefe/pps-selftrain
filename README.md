# pps-selftrain

Lab notebook for reproducing Proxy Policy Steering (PPS, Ning, Wang, Ma and Fang, CoRL 2026, arXiv 2609.09148) and extending it with a self-training loop. This file is written before any results exist. Sections above the log state what we plan to test and how. The log records what actually happened.

PPS keeps a flow matching base policy frozen and trains two small proxies. A reference proxy is distilled from the base on target task observations. A task proxy starts as a copy of the reference and is fine tuned on task demos. At every denoising step the sampler follows

```
v_pps = v_base + gamma * (v_task - v_ref)
```

so the base stays in control wherever the two proxies agree.

## Research question

"When PPS is trained repeatedly on its own successful rollouts, does the reference proxy's imitation of the base degrade on newly visited states, and does re-distilling it on those states preserve recovery?"

## Prediction

Without refresh, the gap between v_ref and v_base grows on states visited in later rounds, so the steering delta (v_task minus v_ref) picks up imitation error and recovery under perturbation declines across rounds. With refresh, the gap stays near round-zero levels and recovery holds. If the gap never grows, refreshing the reference proxy won't matter, and the finding is that the reference proxy is already robust to this drift. This is a hypothesis to test. It is not a result.

## Conditions

Four conditions, all from the same starting demos, with the base frozen throughout.

1. PPS at round zero. Reference proxy distilled from the base, task proxy fine tuned from the reference on the original demos.
2. Task proxy trained longer on the original demos, with gradient updates matched to conditions 3 and 4. This is the matched-data control.
3. Self-training with the reference proxy fixed. Each round, run a fixed budget of steered rollouts, keep the successes, add them to the task proxy's training set, and retrain the task proxy initialized from the fixed reference proxy.
4. Self-training with the reference proxy refreshed. Same as 3, but each round the reference proxy is first re-distilled from the frozen base on a sample of states visited in that round's rollouts (DAgger with the base as expert), and the task proxy is then initialized from the refreshed reference proxy.

In both 3 and 4 the task proxy is initialized from the current reference proxy every round, and this is what keeps the comparison fair. Each round's steering delta starts at zero and grows only from what the task proxy learns, so the one difference between 3 and 4 is whether the reference proxy was refreshed.

## Controls

- Equal gradient updates across conditions 2, 3 and 4.
- Gamma selected once on validation episodes at round zero and fixed across rounds.
- A fixed rollout budget per round, with the number of successes reported.
- Collection, validation and final evaluation use disjoint starting configurations.
- Perturbations applied at a defined task event, such as after grasp.
- 100 to 200 evaluation episodes per condition per round.
- Three seeds minimum.

## Measurements

Recorded every round for every condition.

- Success rate.
- Recovery rate under perturbation.
- Action diversity across successful rollouts.
- Reference proxy fidelity, defined as cosine similarity and MSE between v_ref and v_base on a held-out sample of that round's visited states, logged per denoising step.
- Compute cost of the round, including base forward passes used for refresh.

## Limitations

- The loop is success-only, because the flow-matching loss cannot use failed rollouts. Q-Planning reports that success-only fine-tuning stalled on real tasks.
- Refreshing the reference proxy costs base forward passes each round. That cost will be measured in the pilot.

## Related work and citations

Closest prior work
- ReGuide, Lin et al., 2026, arXiv 2606.28939. Feeds successful guided rollouts back into a diffusion policy across rounds. Updates the policy; we keep the base frozen and update only the proxies.
- Q-Planning, Giridhar et al., 2026, arXiv 2608.21204. Frozen base policy plus a small learned Q-function that improves from both successes and failures. Reports that fine-tuning on successes alone stalls. Our loop is success-only, so this is a known limitation.
- DAgger, Ross et al., AISTATS 2011. Querying the expert on learner-visited states. Condition 4 re-distills the reference proxy from the frozen base on visited states, which is DAgger with the base as expert.
- On-Policy Distillation, Agarwal et al., ICLR 2024, arXiv 2306.13649. Student learns from teacher feedback on its own samples. Same motivation as condition 4, from language models.

Background
- Probe, Learn, Distill, Xiao et al., ICLR 2026, arXiv 2511.00091. Residual RL specialists collect recovery-rich data, then the generalist is fine-tuned on it. We never fine-tune the generalist.
- Self-Imitation Learning, Oh et al., ICML 2018. Learning from an agent's own past successes.

Why we measure diversity
- On-Policy Self-Distillation Reduces Output Diversity, Nicolicioiu et al., 2026, arXiv 2606.26091. Self-distillation can raise average performance while collapsing the variety of successful strategies. We track action diversity every round for this reason.

Method being reproduced
- Proxy Policy Steering, Ning et al., CoRL 2026, arXiv 2609.09148.

## Milestones

0. Find a base checkpoint and a LIBERO task where the base succeeds 10 to 50 percent of the time.
1. PushT toy. A small state based flow matching version of the whole PPS pipeline that runs on a laptop. See docs/toy.md.
2. Full reproduction on pi0.5 with a gamma sweep plot.
3. Self-training loop with the four conditions above.

## Layout

- `toy/` shared code for the PushT toy (data, model, PPS sampler, env, training loop)
- `scripts/toy/` one script per toy step
- `scripts/` future scripts for the full reproduction
- `docs/` notes on each part
- `results/` small result files and plots that are tracked in git, with superseded runs kept under `results/archive/`
- `data/` and `checkpoints/` local only and ignored by git

## Setup

Environment management uses uv. Python is pinned to 3.12 and every dependency is pinned in pyproject.toml and uv.lock.

```bash
uv sync
```

Run the toy end to end, with three proxy seeds. About 25 minutes on an M5 laptop.

```bash
uv run python scripts/toy/train_base.py
```

```bash
uv run python scripts/toy/make_task.py
```

```bash
for s in 0 1 2; do uv run python scripts/toy/distill_ref.py --seed $s && uv run python scripts/toy/train_task.py --seed $s; done
```

```bash
uv run python scripts/toy/eval.py --seeds 0 1 2
```

## Log

### 2026-11-03

Notes from building the PushT toy, limited to what did not go as planned. Numbers for this first version come from results/archive/2026-11-03_shift40, where success means the env's 0.95 threshold.

- Filtering episodes by goal region is impossible in PushT, because every episode shares one goal pose. The variant translates the goal by (40, 40) px instead.
- lerobot/pusht has no block pose in its state, so the toy joins the block keypoints from lerobot/pusht_keypoints, which holds the same 206 episodes.
- Only 7 of 206 PushT demos ever reach the env's 0.95 coverage success threshold. The 50 variant demos are therefore near successes with final coverage of at least 0.9 (mean 0.92, 4 of 50 above 0.95).
- The toy does not show the paper's interior gamma optimum. On the variant the task proxy alone succeeds 4.8 percent of the time, and PPS at gamma 0.25 to 0.75 stays at or below 1.2 percent. Only gamma 1 (6.7 percent) beats the task proxy, in 2 of 3 seeds and tied in the third. That gap is about 1.4 pooled standard errors, so it is not established. A likely reason is that on this variant the base's prior points at the old goal everywhere, so keeping part of the base velocity hurts.
- All success rates on the variant are under 10 percent, so the toy has little resolution for comparing conditions. In a one-off check that is not in the repo, a proxy-sized network trained from scratch on the same 50 demos without translation reached 1 percent on the original task. That suggests the limit is the 50 demos and not a bug in the translation.

### 2026-11-03, variant redesign

What changed and why.

- The 40 px goal shift left the base at zero success with a prior that pointed at the wrong goal everywhere. That is the one setting the paper says PPS should not help, so it was not a fair test. The variant was redesigned so the base is weak but not hopeless.
- Primary metrics are now mean max coverage and success at 0.90, meaning max coverage above 0.90 at any point in the episode. Success at the env's 0.95 threshold stays in every JSON file as success_095. Only 7 of 206 human demos ever reach 0.95, so the 0.95 rates sat near zero and could not separate conditions.
- The base was not retrained. Re-evaluated on the full task with the same 200 episodes as before it gets mean max coverage 0.738, success at 0.90 of 67.5 percent and success at 0.95 of 40.0 percent.
- Goal shift sweep, offset (s, s), base on 200 validation episodes each. 8 px gives coverage 0.707 and success at 0.90 of 11.0 percent. 12 px gives 0.640 and 4.0 percent. 16 px gives 0.568 and 3.0 percent. 20 px gives 0.499 and 3.0 percent. Only 8 px lands inside the 10 to 30 percent band, so it was chosen and the hard initial pose fallback was not needed. Note that 8 px sits near the bottom of the band.
- New demos for the 8 px shift. 194 of 206 translated demos replay within 1 px, 89 of those end at coverage 0.9 or higher, and 50 were sampled (mean final coverage 0.92, all 50 above 0.90, 3 above 0.95).
- Reference and task proxies retrained for seeds 0, 1 and 2. The reference matches the base at cosine 0.994 along the base sampler's path. The task proxy moved only a little from the reference (cosine 0.99, residual norm about 0.67 against 5.96 for the base velocity).
- Gamma sweep extended with 1.25 and 1.5. The plot now has one panel per primary metric. Old results are in results/archive/2026-11-03_shift40.

Results on the variant, 3 proxy seeds by 200 evaluation episodes, same starts and noise for every condition.

| Condition | Mean max coverage | Success at 0.90 |
| --- | --- | --- |
| Base alone | 0.657 | 12.5% |
| Task proxy alone | 0.565 | 30.0% |
| Reference proxy alone | 0.454 | 6.3% |
| PPS gamma 0 | 0.657 | 12.5% |
| PPS gamma 0.25 | 0.693 | 17.7% |
| PPS gamma 0.5 | 0.694 | 23.8% |
| PPS gamma 0.75 | 0.687 | 35.0% |
| PPS gamma 1.0 | 0.691 | 38.7% |
| PPS gamma 1.25 | 0.665 | 39.5% |
| PPS gamma 1.5 | 0.642 | 35.2% |

- No middle gamma peak. Success at 0.90 rises up to gamma 1 and 1.25 and drops at 1.5. Gamma 0.5 is 15.2 ± 2.4 points below gamma 1. Mean max coverage is flat from gamma 0.25 to 1.0 (gamma 0.5 minus gamma 1 is 0.003 ± 0.013) and falls above 1. The peak check in results/toy_gamma_sweep.json finds no middle peak on either metric. Differences here are paired over the same 600 episodes, ± one standard error.
- PPS does beat both of its parts. At gamma 1 it is above the task proxy alone by 0.125 ± 0.018 in coverage and 8.8 ± 2.5 points in success at 0.90, and above the base by 0.033 ± 0.014 and 26.2 ± 2.4 points.
- The two metrics disagree on shape. Coverage rewards any gamma from 0.25 to 1.0 about equally, while success at 0.90 keeps improving up to gamma 1. One reading, not tested here, is that the base already gets the T close and the residual mostly helps with the last few pixels.
- Decision. The toy stops here without a middle gamma peak. Next is milestone 0.

### 2026-11-03, research question revised

The research question, prediction, conditions, controls and milestone 3 were rewritten, and Measurements, Limitations and Related work sections were added. No code or results changed, and no self-training run has happened yet.

- Why. PPS steers with the difference between the task and reference proxies, and the reference proxy is distilled from the base only on the original task observations. Self-training sends rollouts into states that distillation never covered, so any error in the reference proxy's imitation of the base there feeds straight into the steering delta. The old design compared self-training with training longer but could not tell this effect apart from the kind of data being kept. The new design measures reference proxy fidelity every round and adds condition 4, which re-distills the reference proxy on visited states, to test whether this drift matters.
- Two of the three open design decisions listed under the old Conditions section are now settled. Each round the task proxy restarts from the current reference proxy, and steered rollouts do the collecting. The sampling ratio between original demos and kept rollouts is still open.

### 2026-11-04, milestone 0

Milestone 0 is met with pi05_libero on held-out LIBERO-90 tasks. Full numbers are in results/milestone0.md, and raw per-episode records are in results/m0/. Setup is openpi commit 215abfb with the PyTorch backend on an RTX PRO 6000 Blackwell (sm_120), and the local openpi changes are in results/m0/openpi_local.patch. The eval scripts are in scripts/m0/. The server takes flow matching noise from the client, seeded per episode, so every episode is reproducible. A rerun of 5 episodes gave identical outcomes and step counts.

Install problems and fixes.

- torch build. openpi's uv.lock pins torch 2.7.1 from PyPI, which is the CUDA 12.6 build and has no sm_120 kernels. Fixed by adding the PyTorch cu128 index as a uv source for torch and torchvision and re-locking, which gives 2.7.1+cu128 with sm_120. The transformers patch openpi needs was applied with UV_LINK_MODE=copy so it stays out of the uv cache.
- EGL. MuJoCo headless rendering failed with "'NoneType' object has no attribute 'eglQueryString'" because the container had the NVIDIA EGL vendor library but no glvnd loader libEGL.so.1. Fixed with apt-get install libegl1 libgl1 libglvnd0. This lives outside /workspace and has to be redone on a fresh pod.
- LIBERO config. LIBERO asks for a config path interactively on first import. Fixed by writing /workspace/.libero/config.yaml and setting LIBERO_CONFIG_PATH.
- Checkpoint conversion. openpi's JAX to PyTorch converter copies assets from the parent of the checkpoint directory, which does not hold them for the downloaded layout, so the norm stats were missing. Fixed by copying the assets folder into each converted checkpoint by hand.

Two smaller issues. RunPod's nginx holds port 8001, so servers use 8000 and 8002 to 8005. torch.compile max-autotune logs runtime errors for some matmul candidates on sm_120, but inductor skips them and the compiled policy runs fine. All environment variables are in /workspace/env.sh.

What pi05_libero was trained on. The pi05_libero config fine-tunes on physical-intelligence/libero, which openpi builds from libero_10, libero_goal, libero_object and libero_spatial. The dataset's task list has exactly those 40 tasks. LIBERO-90 is excluded entirely. The hold-out is mostly at the scene level, because several LIBERO-90 instructions reuse training wording in a new scene (three are word for word identical). Task 38 is the reverse case, a new instruction in a scene used by a LIBERO-10 training task. Whether pi05_base saw LIBERO in pretraining is not documented.

The screen is bimodal. All 90 LIBERO-90 tasks were run for 5 episodes each, on init states kept apart from the evaluation states. 56 tasks went 0 of 5 and 16 went 5 of 5, and only 9 landed at 1 or 2 of 5. Most held-out tasks are therefore all or nothing for this checkpoint, and the usable middle is small.

Evaluation, 30 episodes each on init states 0 to 29.

| libero_90 task | Success | 95% interval |
| --- | --- | --- |
| 28 close the top drawer of the cabinet | 40.0% | 24.6 to 57.7 |
| 30 put the black bowl on the plate | 70.0% | 52.1 to 83.3 |
| 38 put the right moka pot on the stove | 46.7% | 30.2 to 63.9 |
| 59 pick up the tomato sauce and put it in the tray | 36.7% | 21.9 to 54.5 |
| 64 stack the right bowl on the left bowl and place them in the tray | 53.3% | 36.1 to 69.8 |

- Uncertainty caveat. With 30 episodes each interval spans about 30 points, so the order of tasks 28, 38, 59 and 64 is not established and any could sit a little outside the band. The 5-episode screen was a weak guide, since task 30 went from 2 of 5 to 70 percent.
- Cost. 10 denoising steps by default. A policy call takes 44 ms at batch 1, and an episode takes 3 to 8 s with the GPU to itself. Peak GPU memory is 7.77 GB allocated and 8.11 GB reserved per server.
- Experiments 2 (pi05_base on LIBERO without fine-tuning) and 3 (object placement offsets) were skipped because experiment 1 met the target.
- Decision. Task 59 is the primary task, with task 38 and then task 28 as backups. Tasks 59 and 38 are pick-and-place tasks, so a mid-episode perturbation is natural for the later recovery tests.
