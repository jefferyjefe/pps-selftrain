# Stage 2, PPS plumbing on pi0.5

Stage 2 runs PPS on the real base, pi05_libero with openpi's PyTorch backend, on LIBERO-90 task 59 ("pick up the tomato sauce and put it in the tray", 36.7 percent for the base in milestone 0). This file covers the plumbing that comes before any proxy is trained. It has three parts, a steering capable sampler, the reference proxy distillation dataset and a perturbation wrapper for recovery tests. Every departure from the paper's recipe is listed in the last section.

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
