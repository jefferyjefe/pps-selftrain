# Stage 2, PPS plumbing on pi0.5

Stage 2 runs PPS on the real base, pi05_libero with openpi's PyTorch backend, on LIBERO-90 task 59 ("pick up the tomato sauce and put it in the tray", 36.7 percent for the base in milestone 0). This file covers the plumbing that comes before any proxy is trained. It has three parts, a steering capable sampler, the reference proxy distillation dataset and a perturbation wrapper for recovery tests (part 3 still to come). Every departure from the paper's recipe is listed in the last section.

## Code layout

- `pps_real/sampler.py` is the steering capable sampler. It runs in the openpi venv (Python 3.11).
- `pps_real/policy.py` wraps an openpi Policy so the full input and output transforms run around our sampler.
- `pps_real/libero_obs.py` turns a LIBERO observation into the policy's inputs exactly as the milestone 0 client did. It runs in the LIBERO venv (Python 3.8).
- `scripts/stage2/` holds one script per step. Each script's docstring gives its exact command.

## 1. Steering capable sampler

`pps_real.sampler.sample_actions` is a line by line copy of `PI0Pytorch.sample_actions` at openpi commit 215abfb with two additions. After the base velocity is computed at each of the 10 Euler steps, a hook is called with `(step, x_t, t, v_base, ctx)`. If it returns a tensor, that tensor replaces `v_base` in the Euler update, and if it returns None the base velocity is used unchanged. The second addition is that the final layer prefix hidden states, which the stock sampler computes and throws away, are exposed through `ctx.prefix_out` so the proxies can read the base's own features without a second forward pass. openpi itself is not patched for this. The file calls the model's own submodules, so it tracks the checkpoint without copying any weights or layers.

`pps_hook(v_task_fn, v_ref_fn, gamma)` implements Eq. 4, `v_base + gamma * (v_task - v_ref)`. `Recorder` records `(x_t, t, v_base)` at every step and leaves the velocity alone.

Time follows openpi's convention, which runs the other way from the toy. t = 1 is pure noise and t = 0 is the action, the sampler starts at t = 1 with dt = -0.1, and the 10 steps are taken at t = 1.0, 0.9, ..., 0.1.

### Equivalence check

`scripts/stage2/check_sampler.py` compares our sampler with the stock one on 20 t59 demo observations (one frame from each of demos 0, 2, ..., 38, at spread out phases of the demo) with fixed seeded noise. Results are in `results/stage2/sampler_check.json`.

With torch.compile disabled, every comparison below is bit for bit equal (`torch.equal` on the raw (10, 32) model output, `np.array_equal` on the final 7 dim actions) on all 20 observations.

- Stock sampler run twice, which checks that the reference is deterministic.
- Our sampler with no hook.
- Our sampler with the Recorder hook.
- Our sampler with an identity hook that returns `v_base`.
- Our sampler with a PPS hook whose delta is exactly zero (`v_base + 1.0 * (z - z)`).
- openpi `Policy.infer` against `SteerablePolicy.infer`, which includes the input and output transforms.
- One batched call of all 20 observations, stock against ours.

Two more checks pass. The recorded path satisfies `x_(k+1) = x_k + dt * v_k` exactly, so the recorded tuples are the path the sampler took. A hook that returns `v_base + 0.05` changes the output (by up to 0.027) and the Euler update uses exactly the returned velocity.

### torch.compile

The pi05_libero config wraps the stock `sample_actions` in torch.compile with mode max-autotune, and that is what the milestone 0 server ran. A compiled function cannot be entered halfway to insert a hook, so our path runs eager. The compiled stock sampler does not match the eager stock sampler either. On the same 20 inputs none of the outputs are bit exact, the raw output differs by up to 0.016 (mean 0.0008), and the final actions by up to 0.010 (mean 0.001). Compilation reorders bf16 arithmetic, so exact equality can only be asked of the eager path, and that is the path we verified.

So the base used from here on is the eager sampler, not the compiled one milestone 0 measured. The differences are around one percent of a typical action magnitude, but the base's success rate on t59 has not been re-measured on the eager path. The recovery baselines in part 3 run on the eager path, so they are directly comparable with later PPS runs.

| Sampler, batch 1 | Median ms per call |
| --- | --- |
| Stock, compiled max-autotune | 37.4 |
| Stock, eager | 90.9 |
| Ours, eager, no hook | 89.5 |
| Ours, eager, Recorder hook | 92.0 |
| Ours, full `SteerablePolicy.infer` | 97.9 |
| Stock compiled `Policy.infer` | 42.0 |

The hook itself costs about 2 ms. Losing compilation is what costs, and our hooked path is 2.46 times slower than the compiled stock sampler (92.0 against 37.4 ms). At batch 1 an episode step goes from about 44 to about 98 ms of policy time. If that matters later, the denoise step alone could be compiled, since the hook sits between steps. That was not tried.

Batch size also changes the numbers. The batched call of 20 matches between stock and ours exactly, but differs from the same 20 inputs run one at a time by up to 0.008 in the raw output. This is the same bf16 effect and it matters for part 2.

## 2. Reference proxy distillation dataset

### Recipe

The recipe is the paper's Eq. 5 with Appendix A.3, the same one the toy used (scripts/toy/distill_ref.py). For a demo observation the frozen base sampler is rolled from Gaussian noise through its 10 Euler steps, and at each step the tuple (observation features, x_t, t, v_base) on the base's own path is recorded. t is not sampled at random, so the supervision sits exactly on the noisy actions and noise levels where steering will later act. `scripts/stage2/collect_distill.py` does this with the Recorder hook of part 1, so the recorded path is the exact path of the verified sampler.

### Observation source

The source is every frame of the 50 LIBERO-90 t59 demos from yifengzhu-hf/LIBERO-datasets, 5571 frames in all (95 to 177 per demo). The hdf5 stores only 128 px images, while the base sees 256 px renders resized to 224. So `scripts/stage2/export_demo_states.py` exports the recorded MuJoCo states and `scripts/stage2/render_demo_obs.py` sets each state in the same 256 px environment the eval uses and processes the images exactly as the eval client does (`pps_real/libero_obs.py`). The prompt is the task's own instruction.

One quirk of the LIBERO hdf5 showed up here. The recorded low dimensional observation at index i matches the sim state at index i + 1 (end effector position agrees to 0.5 mm with a shift of one, against 5 to 11 mm without). Rendering from the states sidesteps this, because image and state then come from the same sim state. The remaining pixel difference against the stored 128 px images is a mean of 2.2 of 255 for the agent view and 5.6 for the wrist camera, which is about what the one step offset alone would produce.

### What the proxy sees

The proxy's observation input is the base's own frozen vision language features, not raw pixels. The paper's proxies take the same observations as the base. Feeding them the base's representation of those observations keeps that property without training a new vision encoder, keeps the proxies small (the paper's proxies are around 1 percent of the base), and costs no extra forward pass because the sampler already computes the prefix and part 1 exposes it.

Which features. pi0.5 runs a PaliGemma backbone over the prefix, which is 256 SigLIP tokens per camera slot (agent view, wrist, and a third slot that pi05_libero masks out) plus up to 200 language tokens, 13 of them valid for this prompt. The output is 968 tokens of width 2048 after the final norm. We take the final layer output rather than the SigLIP embeddings because the prefix uses full attention, so these tokens already combine both cameras with the instruction. The action expert attends to the prefix at every layer through its key value cache, so no single tensor is exactly what it reads, and the final layer output is the most processed summary of it. Storing all 525 valid tokens per frame would take 12 GB in bf16, so each camera's 16 by 16 token grid is average pooled to 4 by 4 and the language tokens are averaged, giving 33 tokens of width 2048 per frame (135 KB in float16). The pooling is a choice, not something the paper specifies. It loses spatial detail finer than a quarter of the image, which may matter for the precise grasp phase. Since `policy_obs.npz` keeps every rendered observation, other poolings can be recomputed with one prefix pass per frame (about 3 minutes for all frames) if the proxies need more.

pi05_libero does not read proprioceptive state (discrete_state_input is False and pi0.5 has no state token in the suffix), so the proxy gets none either. The x_t stored are all 32 action dims. LIBERO uses the first 7, but the base denoises all 32 and its velocity depends on all of them, so the proxy has to see the same input.

### Noise draws

The toy draws fresh noise for every training batch, which needs the base live during proxy training. With a 3B base that is too slow, so the dataset fixes 16 noise draws per frame, seeded by `default_rng((59, frame, draw))`. The paper's schedule of 20k steps at batch 64 is 1.28M samples, so with 89k trajectories each one would be seen about 14 times. Whether 16 draws cover the base's action distribution well enough will show up as a gap between training and held out fidelity.

### Batching

Trajectories are collected 64 per sampler call (4 frames by 16 draws), which is 3.6 times faster per trajectory than batch 1 (26.5 against 95 ms). bf16 kernels give slightly different numbers at different batch sizes, so the targets are the base's velocities at batch 64, not at batch 1 as the base runs at eval time. `scripts/stage2/verify_distill.py` measures the gap by replaying sampled (frame, draw) pairs at batch 1. Results are below and in `results/stage2/distill_check.json`.

### Result

The dataset is in `/workspace/data/t59_distill` with a `manifest.json` that records the source, checkpoint, sampler, noise seeds, array shapes, the split marker (demos 45 to 49 held out for fidelity checks, nothing dropped) and the cost.

| | |
| --- | --- |
| Frames | 5571 from 50 demos |
| Trajectories | 89,136 (16 per frame) |
| Tuples (x_t, t, v_base) | 891,360 |
| Size | 3.15 GB (x_t 1.14, v_base 1.14, features 0.75, x0 0.11) |
| GPU time in the sampler | 3166 s (53 min) |
| Wall time of the collection | 3746 s (62 min), with the GPU shared for part of it |
| Base forward passes | 89,136 prefix passes and 891,360 action expert passes, in 1393 calls at batch 64 |
| Rendering the demo frames (before collection) | 124 s |

The prefix is recomputed for every row, so each of a frame's 16 draws pays a full prefix pass. A collector that ran the prefix once per frame and shared its key value cache across the 16 draws would need 5571 prefix passes instead of 89,136. That matters for condition 4, where re-distilling the reference proxy every round is paid for in exactly these passes, but it means stepping outside the sampler verified in part 1, so it was not done here.

Checks (`results/stage2/distill_check.json`).

- Every one of the 89,136 trajectories satisfies `x_(k+1) = x_k + dt * v_k` exactly and ends at its stored x0, and the stored first state of 64 sampled trajectories equals its seeded noise.
- Batch 64 against batch 1, teacher forced, on 64 random (frame, draw) pairs. With the batch 1 base evaluated on the recorded x_t at every step, the per step cosine with the stored v_base is at least 0.99995 and the relative L2 error at most 0.8 percent (median 0.3). The targets are the base's velocities to within bf16 batch noise.
- Batch 64 against batch 1, free replay. When the batch 1 base instead runs its own path from the same noise, 63 of 64 trajectories stay within cosine 0.9999 at every step, but one drifts to cosine 0.967 and its final action chunk differs by up to 0.38. A difference of bf16 rounding at an early step was enough to send that sample somewhere visibly different. So the base's flow is sensitive for a small share of draws, which is worth remembering when PPS deltas, much larger than rounding, are added along the path.
- Features from the batch 1 replay against the stored features have cosine at least 0.989 per token (median 0.998), again the batch size effect in bf16.

## Departures from the paper's recipe

- The base runs the eager sampler, because a compiled function cannot take a per step hook. The compiled sampler that milestone 0 measured differs numerically (actions by up to 0.010).
- Proxies will read the base's frozen final layer prefix features, pooled to 33 tokens, instead of raw observations through their own encoder.
- Distillation targets use 16 fixed noise draws per observation instead of fresh noise every batch.
- Distillation targets were computed at batch 64, so they differ slightly from the base's batch 1 velocities (measured in part 2).
- Distillation observations are the demo states re-rendered at eval resolution, not the hdf5's 128 px images.
