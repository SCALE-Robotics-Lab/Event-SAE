"""openVLA model loading and per-step action inference."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.image_processing_auto import get_image_processor_config

from event_sae.openvla.eval.utils import DEVICE, OPENVLA_V01_SYSTEM_PROMPT


ACTION_DIM = 7


def load_model(cfg: Any):
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.model.checkpoint,
        revision=cfg.model.revision,
        code_revision=cfg.model.code_revision,
        attn_implementation=cfg.model.attn_implementation,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    stats_path = Path(cfg.model.checkpoint) / "dataset_statistics.json"
    if stats_path.is_file():
        model.norm_stats = json.loads(stats_path.read_text())
    return model


def get_processor(cfg: Any):
    # AutoProcessor in Transformers 4.48.1 drops code_revision before loading
    # nested components. Preserve the pin for the custom image processor too.
    processor_config = get_image_processor_config(
        cfg.model.checkpoint, revision=cfg.model.revision,
    )
    processor_class = get_class_from_dynamic_module(
        processor_config["auto_map"]["AutoProcessor"], cfg.model.checkpoint,
        revision=cfg.model.revision, code_revision=cfg.model.code_revision,
    )
    return processor_class.from_pretrained(
        cfg.model.checkpoint,
        revision=cfg.model.revision,
        code_revision=cfg.model.code_revision,
        trust_remote_code=True,
    )


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths], axis=1
    )

    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    if expanded_dims:
        image = image[0]

    return image


def _build_prompt(task_label: str, checkpoint: str) -> str:
    if "openvla-v01" in checkpoint:
        return f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def get_action(model, processor, cfg: Any, obs: dict, task_label: str, unnorm_key: str) -> np.ndarray:
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    if cfg.model.center_crop:
        batch_size = 1
        crop_scale = 0.9
        image_tf = tf.convert_to_tensor(np.array(image))
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
        image_tf = crop_and_resize(image_tf, crop_scale, batch_size)
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        image = Image.fromarray(image_tf.numpy()).convert("RGB")

    prompt = _build_prompt(task_label, cfg.model.checkpoint)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    assert action.shape == (ACTION_DIM,)
    return action
