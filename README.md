# pps-selftrain

Lab notebook for reproducing Proxy Policy Steering (PPS, Ning, Wang, Ma and Fang, CoRL 2026, arXiv 2609.09148) and extending it with a self-training loop. This file is written before any results exist and states what we plan to test and how. What actually happened is recorded in [LOG.md](LOG.md).

PPS keeps a flow matching base policy frozen and trains two small proxies. A reference proxy is distilled from the base on target task observations. A task proxy starts as a copy of the reference and is fine tuned on task demos. At every denoising step the sampler follows

```
v_pps = v_base + gamma * (v_task - v_ref)
```

so the base stays in control wherever the two proxies agree.

## Research question

"When PPS is trained repeatedly on its own successful rollouts, does the reference proxy's imitation of the base degrade on newly visited states, and does re-distilling it on those states preserve recovery?"

## Prediction

Without refresh, the gap between v_ref and v_base grows on states visited in later rounds, so the steering delta (v_task minus v_ref) picks up imitation error and recovery under perturbation declines across rounds. With refresh, the gap stays near round-zero levels and recovery holds. If the gap never grows, refreshing the reference proxy won't matter, and the finding is that the reference proxy is already robust to this drift. This is a hypothesis to test. It is not a result.

Headline prediction against DSRL. Over about 10 rounds, self-training PPS keeps improving while DSRL levels off. DSRL only chooses the noise fed to the frozen base, so it can only reselect behaviors the base already has, while the task proxy can learn new velocity from its kept rollouts. This is also a hypothesis, not a result.

## Scale

10 sim tasks across two benchmarks.

- LIBERO, 4 tasks. t59 (main), t38, t28, and one more from the same milestone 0 screen, still to be chosen.
- RoboCasa, 6 tasks, screened from the existing pi0.5 RoboCasa checkpoint (robocasa365_checkpoints, pi05_pretrain_human300, config pi05_robocasa) for tasks where it succeeds 10 to 50 percent of the time.

Base models are pi0.5 on both benchmarks and pi0 on LIBERO. SmolVLA is an optional third base.

One setup, pi0.5 on LIBERO t59, goes through every stage first. Only after it has run end to end is the pipeline copied to the other tasks and bases.

## Conditions

Five conditions, all from the same starting demos, with the base frozen throughout. The target is about 10 rounds.

1. PPS with no self-training. Reference proxy distilled from the base, task proxy fine tuned from the reference on the original 50 human demos.
2. Task proxy trained longer on the original 50 human demos, with gradient updates matched to conditions 3 and 4. This is the matched-data control.
3. Self-training with the reference proxy fixed. Each round, run a fixed budget of steered rollouts, keep the successes, add them to the task proxy's training set, and retrain the task proxy initialized from the fixed reference proxy.
4. Self-training with the reference proxy refreshed. Same as 3, but each round the reference proxy is first re-distilled from the frozen base on a sample of states visited in that round's rollouts (DAgger with the base as expert), and the task proxy is then initialized from the refreshed reference proxy.
5. DSRL given the same number of robot runs (environment rollouts) as conditions 3 and 4.

In both 3 and 4 the task proxy is initialized from the current reference proxy every round, and this is what keeps the comparison fair. Each round's steering delta starts at zero and grows only from what the task proxy learns, so the one difference between 3 and 4 is whether the reference proxy was refreshed.

Conditions 6 and 7 are defined under Method extension, drift-aware PPS, and depend on the reference proxy actually drifting in conditions 3 and 4.

## Baselines

Scored once against PPS, outside the self-training loop.

- From-scratch specialist. A proxy-sized policy trained only on the 50 task demos.
- Residual policy. A small network trained to add a correction to the frozen base's actions.
- LoRA fine-tuning of the base. This is the priority baseline, because it is the recovery comparison: it adapts the base itself instead of steering it.
- DSRL, using the existing dsrl_pi0 or RLinf code.

All baselines are scored on the same 100 paired trials (same init states and noise) and the same mid-carry drop test as PPS.

## Controls

- Equal gradient updates across conditions 2, 3 and 4.
- Equal environment rollouts across conditions 3, 4 and 5.
- Gamma selected once on validation episodes at round zero and fixed across rounds.
- A fixed rollout budget per round, with the number of successes reported.
- Collection, validation and final evaluation use disjoint starting configurations.
- Perturbations applied at a defined task event, such as after grasp.
- 100 to 200 evaluation episodes per condition per round.
- Three seeds minimum.

## Measurements

Recorded every round for every condition.

- Success rate.
- Recovery rate under the mid-carry drop, where the gripper is forced open while the object is being carried. The exact trigger and strength are set in milestone 1, since the grasp-time release in stage 2 barely hurt the base.
- Action diversity across successful rollouts.
- Reference proxy fidelity, defined as cosine similarity and MSE between v_ref and v_base on a held-out sample of that round's new visited states, logged per denoising step.
- Base forward passes spent per round, for refresh and for rollouts, alongside the round's total compute cost.

## Method extension, drift-aware PPS

This extension only makes sense if the reference proxy actually drifts in conditions 3 and 4. If it does not, it is dropped.

Full refresh (condition 4) queries the base on every sampled state. Drift-aware PPS instead flags states where the reference proxy is likely wrong, without querying the base, using one or more of these signals.

- Novelty of the state relative to the reference proxy's distillation data.
- Disagreement within an ensemble of reference proxies.
- Unusually large corrections, meaning a large v_task minus v_ref.

Two responses on flagged states are tested.

6. Selective refresh. Re-distill the reference proxy from the base only on flagged states.
7. Gated steering. Lower gamma on flagged states, so the base keeps more control where the steering delta is least trusted.

Success means matching full refresh (condition 4) on recovery and success while spending far fewer base forward passes.

## Theory

Bound how the reference proxy's copy error leaks into the steering correction, extending Appendix D of the PPS paper, and check the bound against the drift measured in conditions 3 and 4.

## Stacking (preliminary)

Train a second task proxy from the same reference proxy on 50 demos of a second skill in the same scene. Compare the base, each skill's correction alone, and both corrections added together.

## Real robot

This tests the limitation the PPS paper states itself, that PPS depends on a strong base. The setup is a low-cost SO-101 arm (Seeed Studio SO-ARM101 Pro Assembled Kit) with an overhead USB webcam.

- Fine-tune two pi0.5 bases on SO-101 data, one on about 100 demos and one on about 400, including drop-and-regrab demos, then freeze both.
- On a bottle-to-bowl task, run base vs PPS vs LoRA with about 20 tries each.
- Run the mid-carry drop test.
- Run two or three rounds of self-training with the reference proxy fixed vs refreshed.
- Record side-by-side video.

## Release

Code and a cheap-arm recipe will be open-sourced.

## Limitations

- The loop is success-only, because the flow-matching loss cannot use failed rollouts. Q-Planning reports that success-only fine-tuning stalled on real tasks.
- Refreshing the reference proxy costs base forward passes each round. That cost will be measured in the pilot.

## Related work and citations

Closest prior work
- UF-OPS, Update-Free On-Policy Steering via Verifiers, arXiv 2603.10282. The closest prior work to the self-training loop.
- ReGuide, Lin et al., 2026, arXiv 2606.28939. Feeds successful guided rollouts back into a diffusion policy across rounds. Updates the policy; we keep the base frozen and update only the proxies.
- Q-Planning, Giridhar et al., 2026, arXiv 2608.21204. Frozen base policy plus a small learned Q-function that improves from both successes and failures. Reports that fine-tuning on successes alone stalls. Our loop is success-only, so this is a known limitation.
- DAgger, Ross et al., AISTATS 2011. Querying the expert on learner-visited states. Condition 4 re-distills the reference proxy from the frozen base on visited states, which is DAgger with the base as expert.
- On-Policy Distillation, Agarwal et al., ICLR 2024, arXiv 2306.13649. Student learns from teacher feedback on its own samples. Same motivation as condition 4, from language models.

Other ways to steer a frozen base
- DSRL, Wagenmaker et al., CoRL 2025, arXiv 2506.15799. RL over the noise fed to a frozen diffusion or flow policy. Condition 5 and a baseline.
- Latent Policy Steering, arXiv 2603.05296.
- Flow Reversal Steering, arXiv 2606.13675.
- Flow control steering for VLAs, arXiv 2606.10180.

Composing policies, for the stacking section
- Composable Diffusion, Liu et al., ECCV 2022. Adds the outputs of separately trained diffusion models to combine concepts.
- PoCo, Wang et al., RSS 2024. Composes diffusion policies trained on different data and tasks at sampling time.

Background
- Probe, Learn, Distill, Xiao et al., ICLR 2026, arXiv 2511.00091. Residual RL specialists collect recovery-rich data, then the generalist is fine-tuned on it. We never fine-tune the generalist.
- Self-Imitation Learning, Oh et al., ICML 2018. Learning from an agent's own past successes.

Why we measure diversity
- On-Policy Self-Distillation Reduces Output Diversity, Nicolicioiu et al., 2026, arXiv 2606.26091. Self-distillation can raise average performance while collapsing the variety of successful strategies. We track action diversity every round for this reason.

Method being reproduced
- Proxy Policy Steering, Ning et al., CoRL 2026, arXiv 2609.09148. The venue needs confirming, since the arXiv page lists none.

## Milestones

Done
- PushT toy. A small state based flow matching version of the whole PPS pipeline that runs on a laptop. See docs/toy.md.
- Milestone 0. Find a base checkpoint and a LIBERO task where the base succeeds 10 to 50 percent of the time.

Planned, in order. Each milestone runs on pi0.5 LIBERO t59 first.

1. Reproduce PPS on t59 with pi0.5, with a gamma sweep plot.
2. Baselines, LoRA first.
3. Self-training loop with the five conditions above.
4. Drift-aware PPS (conditions 6 and 7) and the theory bound.
5. Extend across all 10 tasks and every base.
6. Real robot on the SO-101.

Extra. Stacking.

## Layout

- `toy/` shared code for the PushT toy (data, model, PPS sampler, env, training loop)
- `scripts/toy/` one script per toy step
- `scripts/` future scripts for the full reproduction
- `docs/` notes on each part
- `LOG.md` dated lab notebook of what was run and what happened
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
