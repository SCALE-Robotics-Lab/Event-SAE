"""Gemini VLM annotation of task-local event clusters.

For each cluster, sends the prompt (`prompts.build_cluster_annotation_prompt`)
plus the cluster's representative frame sequences (PNG bytes inline) to
Gemini, parses the JSON response into `{phrase, phase}`, and writes one
JSONL row per cluster.

API key: read from `GEMINI_API_KEY` env var, or pass `--api-key-path` to
read from a file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import google.genai as genai
from google.genai import types

from event_sae.events.io import load_jsonl
from event_sae.events.prompts import PHASE_LABELS, PROMPT_VERSION, build_cluster_annotation_prompt


PHASE_LABEL_SET = set(PHASE_LABELS)


def load_api_key(api_key_path: Path | None = None) -> str:
    """Resolve a Gemini API key from env var or file."""
    env = os.environ.get("GEMINI_API_KEY", "").strip()
    if env:
        return env
    if api_key_path is None:
        raise ValueError(
            "GEMINI_API_KEY not set and no --api-key-path provided. "
            "Set GEMINI_API_KEY or pass a path to a text file containing the key."
        )
    api_key_path = Path(api_key_path).resolve()
    if not api_key_path.is_file():
        raise FileNotFoundError(f"Gemini API key file not found: {api_key_path}")
    api_key = api_key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError(f"Gemini API key file is empty: {api_key_path}")
    return api_key


def call_gemini(
    *,
    client: genai.Client,
    model: str,
    prompt: str,
    frame_path_groups: list[list[str]],
    clip_relative_progress_percents: list[float],
    temperature: float = 0.2,
) -> str:
    contents: list[types.Part | str] = [prompt]
    for sequence_idx, frame_paths in enumerate(frame_path_groups, start=1):
        progress_text = ""
        if sequence_idx <= len(clip_relative_progress_percents):
            progress_text = (
                f" This clip occurs at about "
                f"{100.0 * float(clip_relative_progress_percents[sequence_idx - 1]):.1f}% "
                "of its source trajectory."
            )
        contents.append(
            f"BEGIN CLIP {sequence_idx}: the next {len(frame_paths)} PNG images are consecutive "
            f"frames from one short clip in chronological order.{progress_text}"
        )
        for frame_idx, frame_path in enumerate(frame_paths, start=1):
            frame_step_match = re.search(r"step(\d+)", Path(frame_path).name)
            frame_step = frame_step_match.group(1) if frame_step_match is not None else "unknown"
            contents.append(
                f"Clip {sequence_idx}, frame {frame_idx}/{len(frame_paths)}, step {frame_step}:"
            )
            contents.append(
                types.Part.from_bytes(
                    data=Path(frame_path).read_bytes(),
                    mime_type="image/png",
                )
            )
        contents.append(f"END CLIP {sequence_idx}")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
        ),
    )
    return response.text


def parse_annotation_response(response_text: str) -> tuple[str | None, str | None, str | None]:
    """Parse `{phrase, phase}` JSON; returns (phrase, phase, error)."""
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        return None, None, f"json_decode_error: {exc}"
    if not isinstance(parsed, dict):
        return None, None, f"top_level_json_not_object:{type(parsed).__name__}"

    phrase = parsed.get("phrase")
    phase = parsed.get("phase")
    parse_errors: list[str] = []

    if not isinstance(phrase, str) or not phrase.strip():
        phrase = None
        parse_errors.append("missing_or_invalid_phrase")
    else:
        phrase = phrase.strip()

    if not isinstance(phase, str):
        phase = None
        parse_errors.append("missing_or_invalid_phase")
    else:
        phase = phase.strip()
        if phase not in PHASE_LABEL_SET:
            parse_errors.append(f"invalid_phase:{phase}")

    if parse_errors:
        return phrase, phase, "; ".join(parse_errors)
    return phrase, phase, None


def annotate_clusters(
    clusters_path: Path,
    output_path: Path,
    *,
    model: str = "gemini-2.5-flash",
    api_key: str | None = None,
    temperature: float = 0.2,
    max_clusters: int | None = None,
) -> None:
    """Annotate each cluster with Gemini and write one JSONL row per cluster."""
    clusters_path = Path(clusters_path).resolve()
    if not clusters_path.is_file():
        raise FileNotFoundError(f"clusters.jsonl not found: {clusters_path}")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clusters = load_jsonl(clusters_path)
    if max_clusters is not None:
        clusters = clusters[:max_clusters]

    if api_key is None:
        api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    with output_path.open("w", encoding="utf-8") as out_file:
        for idx, cluster in enumerate(clusters, start=1):
            frame_path_groups = cluster["representative_frame_paths"]
            progress_percents = [
                float(p) for p in cluster.get("representative_progress_percents", [])
            ]
            prompt = build_cluster_annotation_prompt(
                task_description=cluster["task_description"],
                cluster_id=cluster["cluster_id"],
                num_sequences=len(frame_path_groups),
                num_frames_per_sequence=len(frame_path_groups[0]) if frame_path_groups else 0,
                episode_coverage=float(cluster["episode_coverage"]),
                relative_progress_percents=progress_percents,
            )
            record = {
                "cluster_id": cluster["cluster_id"],
                "task_id": cluster["task_id"],
                "task_suite_name": cluster["task_suite_name"],
                "task_description": cluster["task_description"],
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "representative_sample_ids": cluster["representative_sample_ids"],
                "representative_clip_paths": cluster["representative_clip_paths"],
                "representative_frame_paths": frame_path_groups,
                "representative_progress_percents": progress_percents,
                "episode_coverage": cluster["episode_coverage"],
            }
            try:
                response_text = call_gemini(
                    client=client,
                    model=model,
                    prompt=prompt,
                    frame_path_groups=frame_path_groups,
                    clip_relative_progress_percents=progress_percents,
                    temperature=temperature,
                )
                record["raw_response"] = response_text
                phrase, phase, parse_error = parse_annotation_response(response_text)
                record["phrase"] = phrase
                record["phase"] = phase
                record["parse_error"] = parse_error
                record["api_error"] = None
            except Exception as exc:
                record["raw_response"] = None
                record["phrase"] = None
                record["phase"] = None
                record["parse_error"] = None
                record["api_error"] = f"{type(exc).__name__}: {exc}"

            out_file.write(json.dumps(record) + "\n")
            out_file.flush()
            print(
                f"[{idx}/{len(clusters)}] cluster_id={cluster['cluster_id']} "
                f"api_error={record['api_error'] is not None} "
                f"parse_error={record['parse_error'] is not None}"
            )
