# Milestone 0 results

Goal. Find a pi0.5 checkpoint and LIBERO task where the unmodified base policy succeeds roughly 10 to 50 percent of the time.

All runs use openpi commit 215abfb with the PyTorch backend, bf16 weights converted from the official JAX checkpoints, torch 2.7.1+cu128 on an RTX PRO 6000 Blackwell (sm_120). Raw per-episode records are in results/m0/. The local openpi changes are in results/m0/openpi_local.patch.

## Protocol

- Client follows openpi examples/libero/main.py. 10 settle steps with a no-op action, replan every 5 actions from a 10-step chunk, 224 px agent and wrist images, max steps 220 (spatial), 280 (object), 300 (goal), 520 (libero_10) and 400 (libero_90).
- Episode k uses LIBERO init state k, env seed 7 + k, and flow matching noise drawn from numpy default_rng((7, k)) on the client and sent to the server. Each episode is reproducible on its own. Rerunning libero_90 task 59 episodes 0 to 4 on a fresh client gave identical success flags and step counts.
- Success is the LIBERO done signal (all goal predicates true) before the step limit.
- Intervals are Wilson 95 percent intervals.
- Peak GPU memory is torch.cuda.max_memory_allocated and max_memory_reserved inside the server process, at batch size 1. nvidia-smi shows about 8.3 GB per server process including the CUDA context.
- Denoising steps is the default num_steps of PI0Pytorch.sample_actions, read from the code at server start. No run overrides it.
- Wall time per episode is reported two ways. "Uncontended" is one client talking to one server with nothing else on the GPU. "Parallel" is what the batch runs saw with 15 clients sharing 5 servers on one GPU, where a policy call takes about 200 ms instead of 44 ms. Episodes whose first policy call triggered torch.compile are left out of the wall time averages.

## Sanity check on a trained task

pi05_libero on libero_spatial task 0 ("pick up the black bowl between the plate and the ramekin and place it on the plate"), init states 0 to 4. 5 of 5 succeeded, median 88 env steps, 2.2 s per episode uncontended after compilation. The first episode took 171 s because torch.compile with mode max-autotune ran on the first call.

## Experiment 1. pi05_libero on held-out LIBERO-90 tasks

What pi05_libero was trained on. The pi05_libero config fine-tunes pi05_base on the LeRobot dataset physical-intelligence/libero. openpi's conversion script builds that dataset from libero_10_no_noops, libero_goal_no_noops, libero_object_no_noops and libero_spatial_no_noops, and the dataset's meta/tasks.jsonl lists exactly those 40 tasks (1693 episodes). LIBERO-90 is not in it. All 90 LIBERO-90 tasks are therefore held out as task instances. Many share scenes, objects or wording with training tasks. For example, LIBERO-10 tasks are built from LIBERO-90 scenes, and libero_90 task 77 has the same instruction as a LIBERO-10 training task in a different scene. What pi05_base saw during pretraining is not documented in openpi, so it is unknown whether any LIBERO data was in the pretraining mix.

Step 1, screen. All 90 LIBERO-90 tasks, 5 episodes each on init states 30 to 34, which are kept disjoint from the evaluation states. Result 16 tasks at 5 of 5, 5 at 4 of 5, 4 at 3 of 5, 4 at 2 of 5, 5 at 1 of 5, and 56 at 0 of 5. Full list in results/m0/exp1_screen_summary.json.

Step 2, evaluation. Five tasks picked from the 1 or 2 of 5 group to cover different skills and scenes, 30 episodes each on init states 0 to 29. Task 77 was skipped because its instruction matches a training instruction word for word.

| libero_90 task | Instruction | Screen (states 30 to 34) | Eval success (states 0 to 29) | 95% interval | In 10 to 50% band |
| --- | --- | --- | --- | --- | --- |
| 28 KITCHEN_SCENE5 | close the top drawer of the cabinet | 2/5 | 12/30 = 40.0% | 24.6 to 57.7 | yes |
| 30 KITCHEN_SCENE5 | put the black bowl on the plate | 2/5 | 21/30 = 70.0% | 52.1 to 83.3 | no |
| 38 KITCHEN_SCENE8 | put the right moka pot on the stove | 1/5 | 14/30 = 46.7% | 30.2 to 63.9 | yes, near the top |
| 59 LIVING_ROOM_SCENE3 | pick up the tomato sauce and put it in the tray | 2/5 | 11/30 = 36.7% | 21.9 to 54.5 | yes |
| 64 LIVING_ROOM_SCENE4 | stack the right bowl on the left bowl and place them in the tray | 2/5 | 16/30 = 53.3% | 36.1 to 69.8 | just above |

Note that the screen with 5 episodes was a rough guide. Task 38 went from 1 of 5 to 47 percent and task 30 from 2 of 5 to 70 percent. With 30 episodes each interval spans roughly 30 points, so the order of tasks 28, 38, 59 and 64 is not established and any of them could sit a little outside the band.

### What "held out" means here

Several LIBERO-90 instructions reuse wording from the training suites in a different scene. Three match a training instruction word for word. Tasks 20 and 44 ("turn on the stove") match a libero_goal task, and task 77 matches a LIBERO-10 task. Many more differ only slightly, for example "put the black bowl on the plate" against libero_goal's "put the bowl on the plate", and "pick up the tomato sauce and put it in the tray" against libero_object's "pick up the tomato sauce and place it in the basket". For these tasks the hold-out is at the scene level, meaning the instruction or a close paraphrase was trained and the scene and layout were not.

The scenes differ among the recommended tasks. Task 59 (LIVING_ROOM_SCENE3) and task 28 (KITCHEN_SCENE5) use scenes that no training task uses. Task 38 (KITCHEN_SCENE8) is the reverse case. Its scene is used by the LIBERO-10 training task "put both moka pots on the stove", so for task 38 the scene was trained and the single-pot instruction was not.

## Experiments 2 and 3

Skipped. Experiment 1 met the milestone 0 target, so pi05_base on LIBERO without fine-tuning (experiment 2) and object placement offsets on a standard task (experiment 3) were not run. scripts/m0/offsets.py is still a stub.

## Recommendation

Primary task libero_90 task 59, "pick up the tomato sauce and put it in the tray", with pi05_libero (36.7 percent, 11 of 30). Backups in order are task 38, "put the right moka pot on the stove" (46.7 percent), then task 28, "close the top drawer of the cabinet" (40.0 percent).

Reasoning. Tasks 59 and 38 are pick-and-place tasks, so a perturbation at a defined mid-episode event, such as knocking the object out of the gripper or moving it after grasp, is natural for the later recovery tests. Task 28 is in the band but is a single articulated motion with no grasp, so it gives a less natural point for a mid-episode perturbation and is the last backup. Task 59 also sits nearer the middle of the band than task 38 and uses a scene that is absent from training. Task 64 (53.3 percent) was not chosen because it sits above the band, and task 30 (70 percent) is well above it.

## Cost measurements

| Run | Checkpoint | Denoising steps | Policy call (batch 1) | Wall per episode, uncontended | Wall per episode, parallel | Peak GPU memory (allocated / reserved) |
| --- | --- | --- | --- | --- | --- | --- |
| Sanity, libero_spatial t0 | pi05_libero | 10 | 45 ms | 2.2 s (success, about 90 steps) | not run | 7.77 / 8.11 GB |
| Exp 1 screen, 90 tasks | pi05_libero | 10 | not measured | not measured | 15 to 55 s | 7.77 / 8.11 GB |
| Exp 1 eval, 5 tasks | pi05_libero | 10 | 44 ms | 3.3 s success (170 steps), 7.4 to 8.1 s failure (410 steps) | 37.0 s mean (29.9 success, 44.0 failure) | 7.77 / 8.11 GB |

One-off costs. Converting a JAX checkpoint to PyTorch bf16 takes about 3 minutes and produces a 7.2 GB safetensors file. Server startup plus the first torch.compile (max-autotune) call takes about 3 minutes, and later servers reuse the inductor cache.
