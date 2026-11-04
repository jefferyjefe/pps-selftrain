"""Stage 2 code for PPS on pi0.5 (openpi PyTorch backend) and LIBERO.

Modules split by interpreter.
- sampler.py, policy.py run in the openpi venv (Python 3.11, torch).
- libero_obs.py, perturb.py run in openpi's examples/libero venv (Python 3.8) and must stay 3.8 compatible.
"""

T59_SUITE = "libero_90"
T59_TASK_ID = 59
T59_NAME = "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray"
T59_DEMO_HDF5 = f"/workspace/data/libero_datasets/libero_90/{T59_NAME}_demo.hdf5"
