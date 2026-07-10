"""Shared config for the vision-layer hijack patch attack (alphabet_soup -> salad_dressing)."""

import os

HOME = os.path.expanduser("~")
RUN_DIR = os.path.join(HOME, "autoresearch/runs/autoresearch-hijack")

USER_TASK = os.environ.get("HIJACK_USER_TASK", "pick up the alphabet soup and place it in the basket")
TARGET_TASK = os.environ.get("HIJACK_TARGET_TASK", "pick up the salad dressing and place it in the basket")

# Digital camera-space patch region in the 224x224 agentview: a band on the empty bottom
# floor -> visible/prominent, does not occlude the graspable objects, inside the 0.9 crop.
PATCH_ROW0, PATCH_COL0, PATCH_H, PATCH_W = 158, 30, 56, 164

STATES_NPZ = os.path.join(RUN_DIR, "states_saladdressing.npz")
PATCH_NPY = os.path.join(RUN_DIR, "patch_v1.npy")
