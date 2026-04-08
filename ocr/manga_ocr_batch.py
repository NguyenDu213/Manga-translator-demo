"""
True batch inference for kha-white/manga-ocr (manga_ocr.MangaOcr).
The stock class only exposes per-image __call__; grouping inputs cuts Python/GPU overhead.
"""
from __future__ import annotations

import os
from typing import List

import torch
from manga_ocr.ocr import post_process
from PIL import Image


def _chunk_size() -> int:
    return max(1, int(os.environ.get("MANGA_OCR_BATCH_CHUNK", "16")))


def run_ocr_batch(mocr, pil_images: List[Image.Image], chunk_size: int | None = None) -> List[str]:
    """
    Run OCR on many PIL images using one model.generate() per chunk.

    Args:
        mocr: manga_ocr.MangaOcr instance (must have .model, .feature_extractor, .tokenizer).
        pil_images: List of PIL.Image (RGB or L).
        chunk_size: Max images per forward pass; default from env MANGA_OCR_BATCH_CHUNK (default 16).

    Returns:
        List of strings, same order as input.
    """
    if not pil_images:
        return []
    if chunk_size is None:
        chunk_size = _chunk_size()

    device = mocr.model.device
    out_all: List[str] = []

    for start in range(0, len(pil_images), chunk_size):
        chunk = pil_images[start : start + chunk_size]
        imgs = [im.convert("L").convert("RGB") for im in chunk]
        batch = mocr.feature_extractor(imgs, return_tensors="pt").pixel_values.to(device)
        with torch.inference_mode():
            generated = mocr.model.generate(batch, max_length=300)
        for row in range(generated.shape[0]):
            raw = mocr.tokenizer.decode(generated[row], skip_special_tokens=True)
            out_all.append(post_process(raw))

    return out_all
