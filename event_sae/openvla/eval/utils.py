"""General utilities for openVLA LIBERO evaluation."""

import os
import random

import numpy as np
import torch

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    orig_low, orig_high = 0.0, 1.0
    action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    action[..., -1] = action[..., -1] * -1.0
    return action


def get_resize_size(model_family: str) -> int:
    if model_family == "openvla":
        return 224
    raise ValueError("Unexpected model family")
