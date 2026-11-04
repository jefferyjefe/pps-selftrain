# PushT toy

A small version of the whole PPS pipeline that runs on a laptop in well under an hour. It exists to check that every piece of the method works and to build intuition for the gamma sweep before moving to pi0.5. Nothing here says anything about how PPS behaves on a real VLA.

## Setup choices

Data. LeRobot PushT, 206 human demos at 10 Hz. lerobot/pusht stores only the agent xy and 96 by 96 images, so the block pose is not in its state vector. A state based policy needs the block, so the loader joins lerobot/pusht_keypoints, which is the same 206 episodes with the 8 corner keypoints of the T added. The loader asserts that index, agent state and actions are identical across the two. Both datasets are pinned to a commit hash in toy/data.py.

Observations. Agent xy plus 16 keypoint coordinates, for the current and the previous frame, so 36 numbers. Pixel coordinates in observations and actions go through the same map `(x - 256) / 256`, so the base and both proxies share one action representation, which is assumption A1 in the paper.

Policy. Flow matching in the paper's convention. The noised chunk is `x_k = k * eps + (1 - k) * a`, with k = 1 pure noise and k = 0 clean. Networks regress the velocity `eps - a`. Training draws k from the shifted Beta(1.5, 1) used by pi0 and the paper. Sampling takes 10 Euler steps from k = 1 to k = 0. Each sample is a chunk of 16 actions, of which 8 are executed before replanning. The paper uses 15 and 8 at 15 Hz. PushT runs at 10 Hz and 16 and 8 is the usual setting there.

Networks. A residual MLP in which the noised chunk is concatenated with a sinusoidal time embedding and passed through an MLP, as in the pi0 action expert, and the observation enters every block. The base has width 512 and about 2.98M parameters. Each proxy has width 256 and about 0.77M parameters, roughly a quarter of the base. In the paper the proxies are about 1 percent of pi0.5. The size gap is what makes the reference proxy an imperfect copy of the base, which is the error the task to reference residual is meant to cancel.

Environment. gym-pusht 0.1.8 with keypoint observations. Episodes stop at 300 steps, or earlier when the T covers more than 95 percent of the goal region, which is the env's own success rule.

Metrics. Two primary metrics are reported everywhere. Mean max coverage is the highest goal coverage reached in an episode, averaged over episodes. Success at 0.90 is the fraction of episodes whose max coverage exceeds 0.90. Success at the env's 0.95 threshold is still saved in every JSON file as success_095, but it is no longer primary, because the human demos themselves rarely reach 0.95 and the 0.95 rates are too close to zero to compare conditions.

Device. MPS when available, with PYTORCH_ENABLE_MPS_FALLBACK set in toy/__init__.py so any op MPS lacks runs on CPU. Every script takes a device flag to force CPU.

## The variant

PushT has one fixed goal pose, (256, 256, pi/4), shared by every episode, so filtering episodes by goal region is not possible. The variant instead translates the goal by a shift of s px along both axes, an offset of (s, s). The base should be weak on the variant but not hopeless, so that its prior still carries skill the task needs. make_task.py tries shifts of 8, 12, 16 and 20 px, measures the base on 200 validation episodes for each, and keeps the smallest shift where base success at 0.90 lands between 10 and 30 percent. All candidates are saved to results/toy_shift_sweep.json.

An earlier version of the toy used a 40 px shift. There the base had zero success and its prior pointed at the wrong goal everywhere, which is the setting the paper says PPS should not help. Those results are kept in results/archive/2026-11-03_shift40.

Demos for the variant are original demos rigidly translated by the same offset. PushT physics has no gravity or floor friction and is translation invariant away from the walls, so a translated demo is a valid demo for the translated goal. make_task.py checks this by replaying every translated demo open loop in the offset goal env and keeping only those whose keypoints match the translated recording within 1 pixel throughout. Replay of the original demos in gym-pusht matches the dataset to about 1e-5 pixels, so this is a strict check.

Only 7 of the 206 original demos ever exceed the env's 0.95 coverage threshold. The median final coverage is about 0.90, because the human demonstrators stopped once the T looked aligned. The variant demo pool therefore keeps translated demos whose final coverage is at least 0.9, and samples 50 of them with a fixed seed. The final evaluation starts from the env's default random reset, which is unchanged by the offset.

## Scripts

All scripts live in scripts/toy and share the toy package. Run them in this order.

1. train_base.py trains the base, the broad generalist, on all 206 demos with the flow matching loss for 60k steps at batch 256, then evaluates it on the original task for 200 episodes. Writes checkpoints/toy/base.pt and results/toy_base_full_task.json.
2. make_task.py runs the shift sweep described above, prints both metrics for the base on each candidate, and saves the 50 variant demos for the chosen shift to data/toy/variant_demos.npz. Writes results/toy_shift_sweep.json and results/toy_variant.json.
3. distill_ref.py distills the reference proxy from the frozen base on the variant demo observations. It follows the paper's recipe, Eq. 5 with Appendix A.3. For each observation the frozen base sampler runs its 10 Euler steps from fresh noise, and the proxy regresses the base velocity at every (x_k, k) the sampler visits. k is not sampled at random, so supervision lands exactly on the states and noise levels where steering will later be applied. 20k steps at batch 64, as in the paper. The flag recipe offpolicy instead noises the demo actions at a random k from the shifted Beta and regresses the base velocity there. That is closer to a plain description of velocity distillation but is not what the paper does, and it is kept only for ablation.
4. train_task.py copies the reference proxy into the task proxy and fine tunes it on the 50 demos with the standard flow matching loss, Eq. 6, for 20k steps at batch 64. It then measures velocity alignment along the base sampler's path, the check in the paper's Appendix C.8, and writes results/toy_alignment_seedN.json.
5. steer.py rolls out any sampler on the variant. Mode pps uses `v_base + gamma * (v_task - v_ref)` at every Euler step, mode base is the frozen base, mode task is the task proxy alone, and mode ref is the reference proxy alone. Gamma 0 in mode pps returns the base velocity exactly. The sampler code is in toy/pps.py and eval.py uses the same code.
6. eval.py evaluates base alone, task proxy alone, reference proxy alone and PPS at gamma 0, 0.25, 0.5, 0.75, 1.0, 1.25 and 1.5 on the variant, 200 episodes each by default, for every proxy seed given. The plot has one panel per primary metric. The JSON also stores per-episode max coverage for every condition and a peak check, described below. Writes results/toy_gamma_sweep.json and results/toy_gamma_sweep.png. The flag plot-only redraws the plot from the saved JSON without running any episodes.

Every condition in eval.py sees the same 200 starting configurations and the same sampler noise, so differences between conditions are not due to luck of the draw in starts or noise.

## Seeds

Starting configurations come from disjoint env seed ranges, defined in toy/common.py.

| Use | First env seed |
| --- | --- |
| Base on the original task | 1000000 |
| make_task.py validation on the variant | 2000000 |
| eval.py final evaluation | 3000000 |
| steer.py ad hoc runs | 4000000 |

Proxy training seeds are separate and select both the network initialisation and the minibatch order.

## What to look for in the plot

- The gamma 0 point must sit exactly on the base line. Same starts, same noise, same velocity. If it does not, something is wrong in the sampler.
- The base line is low by construction, since the base pushes the T toward the old goal, 8 px off in each axis.
- Gamma 1 is not the task proxy alone. It is `v_base + v_task - v_ref`, which only equals `v_task` when the reference matches the base perfectly. The gap between the gamma 1 point and the task proxy line is the effect of the leftover mismatch between base and reference. cos_ref_base in the alignment file says how good the copy is. The paper reports about 0.95.
- A middle peak, where some gamma strictly between 0 and 1 beats both the task proxy line and gamma 1, is the paper's main claim in miniature. The base prior is adding something the 50 demos alone do not provide. The paper found the best gamma between 0.4 and 0.6 and used 0.4 everywhere. eval.py tests this per metric. It finds the best gamma, then takes paired per-episode differences against gamma 1 and against the task proxy alone, pooled over seeds. It calls a middle peak only when the best gamma is strictly between 0 and 1 and both differences exceed two paired standard errors. The result is saved under peak_check.
- Gamma 1.25 and 1.5 over-steer past the task proxy. If the curve keeps rising there, the residual is pointing the right way but the proxies are too timid. If it falls, gamma 1 or below is enough.
- Error bars are the standard deviation across the proxy seeds, which captures proxy training noise but not episode noise. The paired standard errors in peak_check capture episode noise.

## Differences from the paper

- The base is a 3M parameter state based MLP trained on one task, not a VLA, and there is no language input.
- Proxies are about a quarter of the base size rather than 1 percent.
- All networks keep an exponential moving average of their weights and evaluation uses it. The task proxy starts from the reference proxy's averaged weights. The paper does not say whether it averages weights.
- The paper does not give learning rates. The toy uses 3e-4 for the base and the reference proxy and 1e-4 for task fine tuning, with linear warmup and cosine decay.
- The paper's demos are successes. Here they are near successes by the env's own threshold, as explained above.
