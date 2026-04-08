from flask import Flask, render_template, request, redirect, send_file, jsonify
from flask_socketio import SocketIO, emit
import io
import zipfile
import json
import warnings
import os
from concurrent.futures import ThreadPoolExecutor
import sys
import uuid

# JPEG quality for intermediate/preview images (85 is visually identical, 20-30% faster to encode)
JPEG_QUALITY_PREVIEW = 85
# JPEG quality for final download
JPEG_QUALITY_FINAL = 95
import time as time_module

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from detect_bubbles import detect_bubbles, detect_bubbles_batch_normal, MAX_ASPECT_RATIO
from process_bubble import process_bubble, process_bubble_auto, is_dark_bubble, get_bubble_background_color, process_bubble_preserve_gradient
from translator.translator import MangaTranslator
from translator.context_memory import ContextMemory
from add_text import add_text
from manga_ocr import MangaOcr
from ocr.manga_ocr_batch import run_ocr_batch
from PIL import Image
import numpy as np
import base64
import cv2


app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "secret_key")

# Initialize SocketIO with auto-detected async mode
def get_async_mode():
    if getattr(sys, 'frozen', False):
        return 'threading'
    try:
        import eventlet
        return 'eventlet'
    except ImportError:
        pass
    try:
        import gevent
        return 'gevent'
    except ImportError:
        pass
    return 'threading'

socketio = SocketIO(app, cors_allowed_origins="*", async_mode=get_async_mode())

# Control verbose logging (set VERBOSE_LOG=1 to enable debug output)
VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "0") == "1"

# WebSocket progress throttle: skip duplicate percent emits
_ws_progress_cache = {}

def log(msg):
    """Print only if verbose logging is enabled."""
    if VERBOSE_LOG:
        print(msg)

MODEL_PATH = "model/model.pt"

# Default max height for split (1.5x width = landscape-ish ratio)
DEFAULT_SPLIT_HEIGHT_RATIO = 2.0

# Ảnh cao (webtoon) vẫn detect tuần tự + chunk; ảnh thường gom batch YOLO
def _yolo_page_batch_size():
    return max(1, int(os.environ.get("YOLO_DETECT_BATCH_SIZE", "4")))


def _gallery_encode_workers():
    return max(1, int(os.environ.get("GALLERY_ENCODE_WORKERS", "4")))


def _google_translate_page_workers():
    """Dịch Google: song song theo trang (mỗi trang vẫn song song bubble nếu đã bật)."""
    return max(1, int(os.environ.get("GOOGLE_TRANSLATE_PAGE_WORKERS", "2")))

# Global cache for OCR instances
_OCR_CACHE = {
    "manga_ocr": None,
}

def split_long_image(image: np.ndarray, max_height_ratio: float = DEFAULT_SPLIT_HEIGHT_RATIO) -> list:
    """
    Split a long image into multiple shorter chunks.
    
    Args:
        image: Input image as numpy array (H, W, C)
        max_height_ratio: Maximum height/width ratio before splitting.
                          Images taller than width * ratio will be split.
                          
    Returns:
        List of image chunks (numpy arrays). If image doesn't need splitting,
        returns a list with just the original image.
    """
    height, width = image.shape[:2]
    max_height = int(width * max_height_ratio)
    
    # If image is not too tall, return as-is
    if height <= max_height:
        return [image]
    
    # Split into chunks
    chunks = []
    current_y = 0
    chunk_num = 0
    
    while current_y < height:
        # Calculate chunk end position
        chunk_end = min(current_y + max_height, height)
        
        # Extract chunk
        chunk = image[current_y:chunk_end, :].copy()
        chunks.append(chunk)
        
        current_y = chunk_end
        chunk_num += 1
    
    print(f"  Split image ({width}x{height}) into {len(chunks)} chunks")
    return chunks


@app.route("/")
def home():
    return render_template("index.html")


def ocr_bubble_images(mocr, pil_images):
    """OCR nhiều ROI: dùng process_batch nếu có (Chrome Lens), không thì batch generate Manga-OCR."""
    if not pil_images:
        return []
    pb = getattr(mocr, "process_batch", None)
    if callable(pb):
        return pb(pil_images)
    return run_ocr_batch(mocr, pil_images)


def _resize_bubble_roi_and_contour(processed_image, contour, target_w, target_h):
    """
    Khớp ROI với bbox YOLO (ew x eh). Nếu resize ảnh, phải scale contour cùng tỉ lệ —
    không thì add_text() dùng boundingRect sai và chữ lệch trong bubble.
    """
    ph, pw = processed_image.shape[:2]
    if ph == target_h and pw == target_w:
        return processed_image, contour
    out = cv2.resize(
        processed_image, (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )
    if contour is None or len(contour) == 0:
        return out, contour
    sx = target_w / float(pw)
    sy = target_h / float(ph)
    scaled = contour.astype(np.float64) * np.array([[[sx, sy]]], dtype=np.float64)
    return out, np.round(scaled).astype(np.int32)


def process_single_image(image, manga_translator, mocr, selected_translator, selected_font, font_analyzer=None, enable_black_bubble=True):
    """Process a single image and return the translated version.
    
    Optimized with batch translation for Gemini to reduce API calls.
    Supports auto font matching when font_analyzer is provided and selected_font is 'auto'.
    """
    results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
    
    if not results:
        return image
    
    # Phase 1: Collect bubble ROIs, chạy OCR theo batch; xử lý nền bubble (process_bubble_auto)
    bubble_data = []
    pil_for_ocr = []
    first_bubble_image = None  # For font analysis
    
    for result in results:
        # Handle both old format (6 items) and new format (7 items with is_dark_bubble)
        if len(result) >= 7:
            x1, y1, x2, y2, score, class_id, is_dark = result[:7]
        else:
            x1, y1, x2, y2, score, class_id = result[:6]
            is_dark = 0
        
        y1i, y2i, x1i, x2i = int(y1), int(y2), int(x1), int(x2)
        roi_raw = image[y1i:y2i, x1i:x2i].copy()

        if first_bubble_image is None:
            first_bubble_image = roi_raw.copy()

        pil_for_ocr.append(Image.fromarray(roi_raw))

        processed_image, cont, bubble_is_dark, detected_color = process_bubble_auto(
            roi_raw.copy(), force_dark=(is_dark == 1)
        )
        eh, ew = y2i - y1i, x2i - x1i
        processed_image, cont = _resize_bubble_roi_and_contour(processed_image, cont, ew, eh)

        bubble_data.append(
            {
                "detected_image": processed_image,
                "contour": cont,
                "coords": (x1i, y1i, x2i, y2i),
                "is_dark": bubble_is_dark,
                "fill_color": detected_color,
            }
        )

    texts_to_translate = ocr_bubble_images(mocr, pil_for_ocr)

    # Phase 2: Batch translate
    if selected_translator == "gemini" and len(texts_to_translate) > 1:
        # Use batch translation for Gemini
        try:
            if manga_translator._gemini_translator is None:
                from translator.gemini_translator import GeminiTranslator
                api_key = getattr(manga_translator, '_gemini_api_key', None)
                if not api_key:
                    raise ValueError("Gemini API key not provided")
                custom_prompt = getattr(manga_translator, '_gemini_custom_prompt', None)
                manga_translator._gemini_translator = GeminiTranslator(
                    api_key=api_key, 
                    custom_prompt=custom_prompt
                )
            
            translated_texts = manga_translator._gemini_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target
            )
        except Exception as e:
            print(f"Batch translation failed, falling back to single: {e}")
            translated_texts = [manga_translator.translate(t, method=selected_translator) for t in texts_to_translate]
    
    elif selected_translator == "copilot" and len(texts_to_translate) > 1:
        # Use batch translation for Local LLM (Ollama, LM Studio, etc.)
        try:
            if not hasattr(manga_translator, '_local_llm_translator') or manga_translator._local_llm_translator is None:
                from translator.local_llm_translator import LocalLLMTranslator
                copilot_server = getattr(manga_translator, '_copilot_server', 'http://localhost:8080')
                copilot_model = getattr(manga_translator, '_copilot_model', 'gpt-4o')
                copilot_custom_prompt = getattr(manga_translator, '_copilot_custom_prompt', None)
                manga_translator._local_llm_translator = LocalLLMTranslator(
                    server_url=copilot_server,
                    model=copilot_model,
                    custom_prompt=copilot_custom_prompt
                )
                print(f"Local LLM translator initialized: {copilot_server} / {copilot_model}")
            
            translated_texts = manga_translator._local_llm_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target
            )
        except Exception as e:
            print(f"Copilot batch translation failed: {e}")
            translated_texts = texts_to_translate  # Return original on error
    
    else:
        # Single translation for other translators
        # Optimized: Use batch translation if available (e.g. for NLLB)
        translated_texts = manga_translator.translate_batch(texts_to_translate, method=selected_translator)
    
    # Phase 3: Add translated text to bubbles
    # Determine correct font path based on font name
    font_path = get_font_path(selected_font)
    for data, translated_text in zip(bubble_data, translated_texts):
        text_color = (255, 255, 255) if data.get("is_dark", False) else (0, 0, 0)
        add_text(
            data["detected_image"],
            translated_text,
            font_path,
            data["contour"],
            text_color,
        )
        x1b, y1b, x2b, y2b = data["coords"]
        image[y1b:y2b, x1b:x2b] = data["detected_image"]

    return image


def get_font_path(font_name: str) -> str:
    """Get the correct font file path based on font name."""
    # Handle legacy fonts with 'i' suffix
    if font_name in ["animeace_", "arial", "mangat"]:
        return f"fonts/{font_name}i.ttf"
    # Yuki-* fonts use exact name
    elif font_name.startswith("Yuki-") or font_name.startswith("yuki-"):
        return f"fonts/{font_name}.ttf"
    else:
        return f"fonts/{font_name}.ttf"


# --- OCR review (prepare → edit → complete) ---------------------------------
PREPARE_CACHE = {}
PREPARE_JOB_MAX_AGE_SEC = 3600


def _prune_prepare_jobs():
    now = time_module.time()
    dead = [k for k, v in PREPARE_CACHE.items() if now - v.get("ts", 0) > PREPARE_JOB_MAX_AGE_SEC]
    for k in dead:
        PREPARE_CACHE.pop(k, None)


def _emit_progress_ws(phase, current, total, message):
    """Emit WebSocket progress. Throttled: only sends when percent changes."""
    try:
        percent = int((current / max(total, 1)) * 100)
        key = (phase, id(current))
        last = _ws_progress_cache.get(key, -1)
        if percent == last:
            return
        _ws_progress_cache[key] = percent
        socketio.emit(
            "progress",
            {
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
                "percent": percent,
            },
        )
    except Exception:
        pass


def snapshot_originals_jpeg_b64_from_pages(all_pages_data):
    """JPEG base64 của ảnh gốc thật (chưa xóa chữ bubble), theo tên trang."""
    out = {}
    for name, data in all_pages_data.items():
        src = data.get("image_original")
        if src is None:
            src = data["image"]
        _, buf = cv2.imencode(".jpg", src, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_PREVIEW])
        out[name] = base64.b64encode(buf.tobytes()).decode("utf-8")
    return out


def snapshot_originals_jpeg_b64_from_upload_list(all_images):
    out = {}
    for item in all_images:
        _, buf = cv2.imencode(".jpg", item["image"], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_PREVIEW])
        out[item["name"]] = base64.b64encode(buf.tobytes()).decode("utf-8")
    return out


def _encode_one_gallery_result(result, split_long_images, originals_b64):
    """Trả về list entry cho một ảnh đã xử lý (có thể nhiều chunk)."""
    entries = []
    try:
        image = result["image"]
        base_name = result["name"]
        orig_full = originals_b64.get(base_name) if originals_b64 else None

        if split_long_images:
            chunks = split_long_image(image)
        else:
            chunks = [image]

        for i, chunk in enumerate(chunks):
            _, buffer = cv2.imencode(".jpg", chunk, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_PREVIEW])
            encoded_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
            if len(chunks) > 1:
                chunk_name = f"{base_name}_part{i+1}"
            else:
                chunk_name = base_name
            entry = {"name": chunk_name, "data": encoded_image}
            if orig_full:
                entry["original_data"] = orig_full
            entries.append(entry)
    except Exception as e:
        print(f"Error encoding {result.get('name')}: {e}")
    return entries


def encode_gallery_items(processed_results, split_long_images, originals_b64=None):
    """
    Chuẩn hóa payload cho translate.html: name, data (đã dịch), original_data (ảnh gốc).
    Nhiều ảnh: encode JPEG/base64 song song (GALLERY_ENCODE_WORKERS).
    """
    if not processed_results:
        return []
    nw = _gallery_encode_workers()
    if len(processed_results) == 1 or nw <= 1:
        out = []
        for result in processed_results:
            out.extend(_encode_one_gallery_result(result, split_long_images, originals_b64))
        return out

    with ThreadPoolExecutor(max_workers=min(nw, len(processed_results))) as ex:
        nested = list(
            ex.map(
                lambda r: _encode_one_gallery_result(r, split_long_images, originals_b64),
                processed_results,
            )
        )
    return [entry for sub in nested for entry in sub]


def _image_is_long_yolo_slice(image) -> bool:
    h, w = image.shape[:2]
    return (h / max(w, 1)) > MAX_ASPECT_RATIO


def _fill_one_page_from_detections(
    name, image, results, all_pages_data, all_bubble_images, bubble_mapping
):
    """Ghi một trang vào all_pages_data + gom ROI cho OCR batch."""
    if not results:
        all_pages_data[name] = {
            "image": image,
            "image_original": image.copy(),
            "bubbles": [],
            "texts": [],
        }
        print(" - 0 bubbles")
        return

    image_original = image.copy()
    print(f" - {len(results)} bubbles")
    bubble_data = []

    for bubble_idx, result in enumerate(results):
        if len(result) >= 7:
            x1, y1, x2, y2, score, class_id, is_dark = result[:7]
        else:
            x1, y1, x2, y2, score, class_id = result[:6]
            is_dark = 0

        y1i, y2i, x1i, x2i = int(y1), int(y2), int(x1), int(x2)
        roi_raw = image[y1i:y2i, x1i:x2i].copy()
        all_bubble_images.append(Image.fromarray(roi_raw))
        bubble_mapping.append((name, bubble_idx))

        processed_image, cont, bubble_is_dark, detected_color = process_bubble_auto(
            roi_raw.copy(), force_dark=(is_dark == 1)
        )
        eh, ew = y2i - y1i, x2i - x1i
        processed_image, cont = _resize_bubble_roi_and_contour(processed_image, cont, ew, eh)
        image[y1i:y2i, x1i:x2i] = processed_image

        bubble_data.append(
            {
                "detected_image": processed_image,
                "contour": cont,
                "coords": (x1i, y1i, x2i, y2i),
                "is_dark": bubble_is_dark,
                "fill_color": detected_color,
            }
        )

    all_pages_data[name] = {
        "image": image,
        "image_original": image_original,
        "bubbles": bubble_data,
        "texts": [],
    }


def detect_and_ocr_only(images_data, mocr, enable_black_bubble=True):
    """
    Chỉ detect bubble + OCR. Trả về all_pages_data giống pipeline batch.
    Ảnh thường (không quá dài) gom batch YOLO; webtoon dài vẫn detect từng ảnh + chunk.
    """
    total_images = len(images_data)

    _emit_progress_ws("detection", 0, total_images, "Bắt đầu phát hiện speech bubbles...")
    all_pages_data = {}
    all_bubble_images = []
    bubble_mapping = []

    yolo_batch = _yolo_page_batch_size()
    processed_count = 0
    i = 0
    while i < len(images_data):
        img_data = images_data[i]
        image = img_data["image"]
        name = img_data["name"]

        if _image_is_long_yolo_slice(image):
            processed_count += 1
            _emit_progress_ws("detection", processed_count, total_images, f"Phát hiện bubbles: {name}")
            print(f"  [{processed_count}/{total_images}] {name}", end="", flush=True)
            results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
            _fill_one_page_from_detections(
                name, image, results, all_pages_data, all_bubble_images, bubble_mapping
            )
            i += 1
            continue

        batch_names = [name]
        batch_images = [image]
        i += 1
        while i < len(images_data) and len(batch_images) < yolo_batch:
            nd = images_data[i]
            im2 = nd["image"]
            if _image_is_long_yolo_slice(im2):
                break
            batch_names.append(nd["name"])
            batch_images.append(im2)
            i += 1

        dets_list = detect_bubbles_batch_normal(MODEL_PATH, batch_images, enable_black_bubble)
        for n, img, results in zip(batch_names, batch_images, dets_list):
            processed_count += 1
            _emit_progress_ws("detection", processed_count, total_images, f"Phát hiện bubbles: {n}")
            print(f"  [{processed_count}/{total_images}] {n}", end="", flush=True)
            _fill_one_page_from_detections(n, img, results, all_pages_data, all_bubble_images, bubble_mapping)

    _emit_progress_ws(
        "detection",
        total_images,
        total_images,
        f"Phát hiện xong {len(all_bubble_images)} bubbles",
    )

    if all_bubble_images:
        _emit_progress_ws("ocr", 0, 1, f"Đang OCR {len(all_bubble_images)} bubbles...")
        print(f"\n[OCR] processing {len(all_bubble_images)} bubbles (batch)...", end=" ", flush=True)
        all_texts = ocr_bubble_images(mocr, all_bubble_images)

        for (page_name, bubble_idx), text in zip(bubble_mapping, all_texts):
            all_pages_data[page_name]["texts"].append(text)
        print("done")
        _emit_progress_ws("ocr", 1, 1, f"OCR hoàn tất ({len(all_bubble_images)} bubbles)")

    return all_pages_data


def translate_pages_gemini_copilot_batch(
    all_pages_data,
    manga_translator,
    translator_type,
    batch_size=10,
    use_context_memory=True,
    emit_ws=False,
):
    """
    Chỉ dịch (Gemini / Local LLM, batch theo trang). Trả về {tên_trang: [chuỗi dịch, ...]}.
    Trang không có bubble: [].
    """
    pages_texts = {name: data["texts"] for name, data in all_pages_data.items() if data["texts"]}
    all_translations = {name: [] for name in all_pages_data}

    if not pages_texts:
        return all_translations

    translator = None
    translator_name = "Unknown"
    if translator_type == "copilot" and getattr(manga_translator, "_local_llm_translator", None):
        translator = manga_translator._local_llm_translator
        translator_name = "Local LLM"
    elif translator_type == "gemini" and getattr(manga_translator, "_gemini_translator", None):
        translator = manga_translator._gemini_translator
        translator_name = "Gemini"

    if not translator:
        if emit_ws:
            _emit_progress_ws("translation", 1, 1, "Dịch hoàn tất")
        return all_translations

    start_phase = time_module.time()
    if emit_ws:
        _emit_progress_ws("translation", 0, 1, "Đang dịch...")
    print(f"{translator_name} batch translating {len(pages_texts)} pages in chunks of {batch_size}...")
    context_memory = ContextMemory() if use_context_memory else None
    if use_context_memory:
        print("  Context Memory enabled - tracking terms and story context")

    page_names = list(pages_texts.keys())
    for i in range(0, len(page_names), batch_size):
        batch_names = page_names[i : i + batch_size]
        batch_texts = {name: pages_texts[name] for name in batch_names}
        print(f"  Translating batch {i//batch_size + 1}: pages {i+1}-{min(i+batch_size, len(page_names))}")
        try:
            translated = translator.translate_pages_batch(
                batch_texts,
                source=manga_translator.source,
                target=manga_translator.target,
                context_memory=context_memory,
            )
            for name, tlist in translated.items():
                all_translations[name] = tlist if isinstance(tlist, list) else list(tlist)
            if context_memory:
                context_memory.update_from_translation(batch_texts, translated)
                stats = context_memory.get_stats()
                print(
                    f"    Context updated: {stats['tracked_words']} terms tracked, {stats['recent_pages']} pages in memory"
                )
        except Exception as e:
            print(f"  Batch failed: {e}, falling back to individual translation")
            for name, texts in batch_texts.items():
                try:
                    all_translations[name] = translator.translate_batch(
                        texts, manga_translator.source, manga_translator.target
                    )
                except Exception:
                    all_translations[name] = list(texts)

    for name, data in all_pages_data.items():
        texts = data["texts"]
        n_b = len(texts)
        tlist = all_translations.get(name, [])
        if n_b == 0:
            all_translations[name] = []
        elif len(tlist) != n_b:
            all_translations[name] = list(texts)

    print(f"✓ Translation (batch) completed in {time_module.time() - start_phase:.1f}s")
    if emit_ws:
        _emit_progress_ws("translation", 1, 1, "Dịch hoàn tất")
    return all_translations


def translate_all_pages_preview(
    all_pages_data,
    manga_translator,
    selected_translator,
    batch_size=10,
    use_context_memory=True,
    emit_ws=False,
):
    """Dịch toàn bộ text OCR để xem demo (không render ảnh)."""
    if selected_translator in ("gemini", "copilot"):
        return translate_pages_gemini_copilot_batch(
            all_pages_data,
            manga_translator,
            selected_translator,
            batch_size=batch_size,
            use_context_memory=use_context_memory,
            emit_ws=emit_ws,
        )
    items = list(all_pages_data.items())
    pw = _google_translate_page_workers()
    if selected_translator == "google" and pw > 1 and len(items) > 1:

        def _preview_one(it):
            name, data = it
            texts = data["texts"]
            if not texts:
                return name, []
            try:
                return name, _translate_bubble_texts_single_page(
                    texts, manga_translator, selected_translator
                )
            except Exception as e:
                print(f"Preview translate failed for page {name}: {e}")
                return name, list(texts)

        with ThreadPoolExecutor(max_workers=min(pw, len(items))) as ex:
            pairs = list(ex.map(_preview_one, items))
        return dict(pairs)

    out = {}
    for name, data in items:
        texts = data["texts"]
        if not texts:
            out[name] = []
            continue
        try:
            out[name] = _translate_bubble_texts_single_page(texts, manga_translator, selected_translator)
        except Exception as e:
            print(f"Preview translate failed for page {name}: {e}")
            out[name] = list(texts)
    return out


def translation_and_render_gemini_copilot(
    all_pages_data,
    manga_translator,
    selected_font,
    translator_type,
    batch_size=10,
    use_context_memory=True,
    total_images=None,
):
    """Phase dịch + render cho Gemini / Local LLM (batch theo trang)."""
    if total_images is None:
        total_images = len(all_pages_data)

    all_translations = translate_pages_gemini_copilot_batch(
        all_pages_data,
        manga_translator,
        translator_type,
        batch_size=batch_size,
        use_context_memory=use_context_memory,
        emit_ws=True,
    )

    _emit_progress_ws("rendering", 0, total_images, "Đang render text vào ảnh...")
    processed_results = []
    font_path = get_font_path(selected_font)
    render_idx = 0
    for name, data in all_pages_data.items():
        render_idx += 1
        _emit_progress_ws("rendering", render_idx, total_images, f"Render text: {name}")
        image = data["image"]
        bubbles = data["bubbles"]
        translated_texts = all_translations.get(name) or []
        if len(translated_texts) != len(bubbles):
            translated_texts = data["texts"]
        for bubble, text in zip(bubbles, translated_texts):
            x1, y1, x2, y2 = bubble["coords"]
            bubble_region = image[y1:y2, x1:x2]
            text_color = (255, 255, 255) if bubble.get("is_dark", False) else (0, 0, 0)
            add_text(bubble_region, text, font_path, bubble["contour"], text_color)
        processed_results.append({"image": image, "name": name})

    _emit_progress_ws("done", total_images, total_images, "Hoàn tất")
    return processed_results


def _translate_bubble_texts_single_page(texts_to_translate, manga_translator, selected_translator):
    """Giống nhánh dịch trong process_single_image (một trang)."""
    if not texts_to_translate:
        return []

    if selected_translator == "gemini" and len(texts_to_translate) > 1:
        try:
            if getattr(manga_translator, "_gemini_translator", None) is None:
                from translator.gemini_translator import GeminiTranslator

                api_key = getattr(manga_translator, "_gemini_api_key", None)
                if not api_key:
                    raise ValueError("Gemini API key not provided")
                custom_prompt = getattr(manga_translator, "_gemini_custom_prompt", None)
                manga_translator._gemini_translator = GeminiTranslator(
                    api_key=api_key, custom_prompt=custom_prompt
                )
            return manga_translator._gemini_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target,
            )
        except Exception as e:
            print(f"Batch translation failed, falling back to single: {e}")
            return [
                manga_translator.translate(t, method=selected_translator) for t in texts_to_translate
            ]

    if selected_translator == "copilot" and len(texts_to_translate) > 1:
        try:
            if not hasattr(manga_translator, "_local_llm_translator") or manga_translator._local_llm_translator is None:
                from translator.local_llm_translator import LocalLLMTranslator

                copilot_server = getattr(manga_translator, "_copilot_server", "http://localhost:8080")
                copilot_model = getattr(manga_translator, "_copilot_model", "gpt-4o")
                copilot_custom_prompt = getattr(manga_translator, "_copilot_custom_prompt", None)
                manga_translator._local_llm_translator = LocalLLMTranslator(
                    server_url=copilot_server,
                    model=copilot_model,
                    custom_prompt=copilot_custom_prompt,
                )
            return manga_translator._local_llm_translator.translate_batch(
                texts_to_translate,
                source=manga_translator.source,
                target=manga_translator.target,
            )
        except Exception as e:
            print(f"Copilot batch translation failed: {e}")
            return texts_to_translate

    return manga_translator.translate_batch(texts_to_translate, method=selected_translator)


def translation_and_render_other_translators(
    all_pages_data, manga_translator, selected_translator, selected_font, total_images=None
):
    """Dịch từng trang + render (NLLB, Google, Bing, …)."""
    if total_images is None:
        total_images = len(all_pages_data)

    _emit_progress_ws("translation", 0, max(total_images, 1), "Đang dịch...")
    items = list(all_pages_data.items())
    pw = _google_translate_page_workers()
    if selected_translator == "google" and pw > 1 and len(items) > 1:

        def _tr_page(it):
            name, data = it
            texts = data["texts"]
            if not texts:
                return name, []
            return name, _translate_bubble_texts_single_page(
                texts, manga_translator, selected_translator
            )

        with ThreadPoolExecutor(max_workers=min(pw, len(items))) as ex:
            pairs = list(ex.map(_tr_page, items))
        all_translations = dict(pairs)
    else:
        all_translations = {}
        idx = 0
        for name, data in items:
            idx += 1
            texts = data["texts"]
            if not texts:
                all_translations[name] = []
            else:
                all_translations[name] = _translate_bubble_texts_single_page(
                    texts, manga_translator, selected_translator
                )
            _emit_progress_ws("translation", idx, max(total_images, 1), f"Dịch: {name}")

    _emit_progress_ws("translation", total_images, max(total_images, 1), "Dịch hoàn tất")

    _emit_progress_ws("rendering", 0, total_images, "Đang render text vào ảnh...")
    font_path = get_font_path(selected_font)
    processed_results = []
    render_idx = 0
    for name, data in all_pages_data.items():
        render_idx += 1
        _emit_progress_ws("rendering", render_idx, total_images, f"Render text: {name}")
        image = data["image"]
        bubbles = data["bubbles"]
        translated_texts = all_translations.get(name, data["texts"])
        for bubble, text in zip(bubbles, translated_texts):
            x1, y1, x2, y2 = bubble["coords"]
            bubble_region = image[y1:y2, x1:x2]
            text_color = (255, 255, 255) if bubble.get("is_dark", False) else (0, 0, 0)
            add_text(bubble_region, text, font_path, bubble["contour"], text_color)
        processed_results.append({"image": image, "name": name})

    _emit_progress_ws("done", total_images, total_images, "Hoàn tất")
    return processed_results


def process_images_with_batch(images_data, manga_translator, mocr, selected_font, translator_type, batch_size=10, use_context_memory=True, enable_black_bubble=True):
    """
    Process multiple images with multi-page batching for Copilot or Gemini.
    Collects all texts first, batch translates, then applies translations.
    """
    total_images = len(images_data)
    log(f"Processing {total_images} images... Context Memory: {'ON' if use_context_memory else 'OFF'}")
    start_time = time_module.time()
    print("\n[Phase 1-2] Detecting bubbles + OCR...")
    all_pages_data = detect_and_ocr_only(images_data, mocr, enable_black_bubble)
    print(f"✓ Detection + OCR finished in {time_module.time() - start_time:.1f}s")
    print("\n[Phase 3-4] Translation + render...")
    processed_results = translation_and_render_gemini_copilot(
        all_pages_data,
        manga_translator,
        selected_font,
        translator_type,
        batch_size=batch_size,
        use_context_memory=use_context_memory,
        total_images=total_images,
    )
    total_time = time_module.time() - start_time
    print(f"{'='*50}")
    print(f"✓ TOTAL: {total_images} images in {total_time:.1f}s ({total_time/max(total_images,1):.1f}s/image)")
    print(f"{'='*50}\n")
    return processed_results


def parse_translate_form(req):
    """Đọc form dịch giống nhau cho /translate và /prepare-ocr. Trả None nếu thiếu file."""
    translator_map = {
        "Opus-mt model": "hf",
        "NLLB": "nllb",
        "Gemini": "gemini",
        "Google": "google",
        "Local LLM": "copilot",
    }
    selected_translator = translator_map.get(
        req.form["selected_translator"],
        req.form["selected_translator"].lower(),
    )
    copilot_server = req.form.get("copilot_server", "http://localhost:8080")
    copilot_model = req.form.get("copilot_model_input", "gpt-4o")
    gemini_api_key = req.form.get("gemini_api_key", "").strip()
    use_context_memory = req.form.get("context_memory") == "on"
    enable_black_bubble = req.form.get("detect_black_bubbles") == "on"
    split_long_images = req.form.get("split_long_images") == "on"
    selected_font_raw = req.form["selected_font"]
    selected_font = selected_font_raw.lower()
    if selected_font == "auto (match original)":
        selected_font = "auto"
    elif selected_font == "animeace":
        selected_font = "animeace_"
    elif selected_font_raw.startswith("Yuki-"):
        selected_font = selected_font_raw
    selected_ocr = req.form.get("selected_ocr", "Manga-OCR").lower()
    source_lang_map = {
        "japanese (manga)": "ja",
        "chinese (manhua)": "zh",
        "korean (manhwa)": "ko",
        "english (comic)": "en",
    }
    selected_source = req.form.get("selected_source_lang", "Japanese (Manga)").lower()
    source_lang = source_lang_map.get(selected_source, "ja")
    target_lang_map = {
        "english": "en",
        "vietnamese": "vi",
        "chinese": "zh",
        "korean": "ko",
        "thai": "th",
        "indonesian": "id",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "russian": "ru",
    }
    selected_language = req.form.get("selected_language", "Vietnamese").lower()
    target_lang = target_lang_map.get(selected_language, "vi")
    style_map = {
        "default": "",
        "casual (thân mật)": "casual",
        "formal (trang trọng)": "formal",
        "keep honorifics (-san, senpai...)": "keep_honorifics",
        "web novel style": "web_novel",
        "action (ngắn gọn)": "action",
        "literal (sát nghĩa)": "literal",
        "custom...": "",
    }
    selected_style = req.form.get("selected_style", "Default").lower()
    style = style_map.get(selected_style, "")
    custom_prompt = req.form.get("custom_prompt", "").strip()
    if custom_prompt:
        style = custom_prompt
    files = req.files.getlist("files")
    if not files or files[0].filename == "":
        return None
    return {
        "selected_translator": selected_translator,
        "copilot_server": copilot_server,
        "copilot_model": copilot_model,
        "gemini_api_key": gemini_api_key,
        "use_context_memory": use_context_memory,
        "enable_black_bubble": enable_black_bubble,
        "split_long_images": split_long_images,
        "selected_font": selected_font,
        "selected_font_raw": selected_font_raw,
        "selected_ocr": selected_ocr,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "style": style,
        "files": files,
    }


def apply_ocr_edits_from_form(req, all_pages_data):
    """Cập nhật all_pages_data['texts'] từ form chỉnh OCR (page_name_{pi}, bubble_count_{pi}, ocr_{pi}_{bi})."""
    pi = 0
    while f"page_name_{pi}" in req.form:
        name = req.form[f"page_name_{pi}"]
        try:
            n = int(req.form.get(f"bubble_count_{pi}", "0"))
        except ValueError:
            n = 0
        if name in all_pages_data:
            texts = [req.form.get(f"ocr_{pi}_{bi}", "") for bi in range(n)]
            all_pages_data[name]["texts"] = texts
        pi += 1


@app.route("/translate", methods=["POST"])
def upload_file():
    cfg = parse_translate_form(request)
    if cfg is None:
        return redirect("/")

    selected_translator = cfg["selected_translator"]
    copilot_server = cfg["copilot_server"]
    copilot_model = cfg["copilot_model"]
    gemini_api_key = cfg["gemini_api_key"]
    use_context_memory = cfg["use_context_memory"]
    enable_black_bubble = cfg["enable_black_bubble"]
    split_long_images = cfg["split_long_images"]
    selected_font = cfg["selected_font"]
    source_lang = cfg["source_lang"]
    target_lang = cfg["target_lang"]
    style = cfg["style"]
    files = cfg["files"]
    
    # Initialize translator and OCR once for all images
    manga_translator = MangaTranslator(source=source_lang, target=target_lang)
    
    # Set custom prompt for Gemini
    if selected_translator == "gemini" and style:
        manga_translator._gemini_custom_prompt = style
    
    # Set custom prompt for Local LLM
    if selected_translator == "copilot" and style:
        manga_translator._copilot_custom_prompt = style
    
    # Set Gemini API key
    if selected_translator == "gemini" and gemini_api_key:
        manga_translator._gemini_api_key = gemini_api_key
        print(f"Using Gemini API with provided key")
    
    # Set Copilot settings
    if selected_translator == "copilot":
        manga_translator._copilot_server = copilot_server
        manga_translator._copilot_model = copilot_model
        print(f"Using Local LLM: {copilot_server} / model: {copilot_model}")
    
    if _OCR_CACHE["manga_ocr"] is None:
        _OCR_CACHE["manga_ocr"] = MangaOcr()
    mocr = _OCR_CACHE["manga_ocr"]
    
    # Initialize font analyzer for auto font matching
    font_analyzer = None
    if selected_font == "auto":
        try:
            from font_analyzer import FontAnalyzer
            # Use same API key as Gemini translator
            api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Warning: No Gemini API key provided for font analysis")
            font_analyzer = FontAnalyzer(api_key=api_key)
            print("Font analyzer initialized for auto font matching")
        except Exception as e:
            print(f"Failed to initialize font analyzer: {e}")
            selected_font = "animeace_"  # Fallback to default
    
    # Process all images
    processed_images = []
    auto_font_determined = False  # Flag to analyze font only once
    
    # For Local LLM and Gemini: Use multi-page batch processing
    if selected_translator in ["copilot", "gemini"]:
        # First, read all images into memory
        all_images = []
        for file in files:
            if file and file.filename:
                try:
                    file_stream = file.stream
                    file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        continue
                    
                    name = os.path.splitext(file.filename)[0]
                    all_images.append({'image': image, 'name': name})
                except Exception as e:
                    print(f"Error reading {file.filename}: {e}")
        
        if not all_images:
            return redirect("/")
        
        # Auto font: analyze first image
        if selected_font == "auto" and font_analyzer is not None:
            try:
                results = detect_bubbles(MODEL_PATH, all_images[0]['image'], enable_black_bubble)
                if results:
                    det = results[0]
                    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                    first_bubble = all_images[0]['image'][int(y1):int(y2), int(x1):int(x2)]
                    selected_font = font_analyzer.analyze_and_match(first_bubble)
                    print(f"Auto font matched: {selected_font}")
                else:
                    selected_font = "animeace_"
            except Exception as e:
                print(f"Font analysis failed: {e}")
                selected_font = "animeace_"
        
        # Initialize translator based on type
        if selected_translator == "copilot":
            if not hasattr(manga_translator, '_local_llm_translator') or manga_translator._local_llm_translator is None:
                from translator.local_llm_translator import LocalLLMTranslator
                # Get custom prompt for Local LLM
                copilot_custom_prompt = style if style else None
                manga_translator._local_llm_translator = LocalLLMTranslator(
                    server_url=copilot_server,
                    model=copilot_model,
                    custom_prompt=copilot_custom_prompt
                )
                print(f"Local LLM translator initialized: {copilot_server} / {copilot_model} (style: {style or 'default'})")
        
        elif selected_translator == "gemini":
            if not hasattr(manga_translator, '_gemini_translator') or manga_translator._gemini_translator is None:
                from translator.gemini_translator import GeminiTranslator
                api_key = gemini_api_key
                if not api_key:
                    raise ValueError("Gemini API key required. Please enter it in the web form.")
                custom_prompt = getattr(manga_translator, '_gemini_custom_prompt', None)
                manga_translator._gemini_translator = GeminiTranslator(
                    api_key=api_key,
                    custom_prompt=custom_prompt
                )
                print("Gemini translator initialized for multi-page batching")
        
        # Process with multi-page batching (10 pages per API call)
        originals_b64 = snapshot_originals_jpeg_b64_from_upload_list(all_images)
        processed_results = process_images_with_batch(
            all_images, manga_translator, mocr, selected_font,
            translator_type=selected_translator, batch_size=10,
            use_context_memory=use_context_memory,
            enable_black_bubble=enable_black_bubble
        )
        processed_images = encode_gallery_items(
            processed_results, split_long_images, originals_b64
        )
    
    else:
        # For other translators: Use per-image processing (original flow)
        for file in files:
            if file and file.filename:
                try:
                    # Read image
                    file_stream = file.stream
                    file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        continue
                    
                    # Auto font: analyze FIRST image only
                    if selected_font == "auto" and font_analyzer is not None and not auto_font_determined:
                        try:
                            results = detect_bubbles(MODEL_PATH, image, enable_black_bubble)
                            if results:
                                det = results[0]
                                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                                first_bubble = image[int(y1):int(y2), int(x1):int(x2)]
                                selected_font = font_analyzer.analyze_and_match(first_bubble)
                                print(f"Auto font matched (once for all images): {selected_font}")
                            else:
                                selected_font = "animeace_"
                        except Exception as e:
                            print(f"Font analysis failed: {e}")
                            selected_font = "animeace_"
                        auto_font_determined = True
                    
                    # Get original filename
                    name = os.path.splitext(file.filename)[0]
                    
                    _, obuf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_PREVIEW])
                    orig_b64_single = base64.b64encode(obuf.tobytes()).decode("utf-8")
                    processed_image = process_single_image(
                        image, manga_translator, mocr,
                        selected_translator, selected_font, None,
                        enable_black_bubble=enable_black_bubble
                    )
                    processed_images.extend(
                        encode_gallery_items(
                            [{"name": name, "image": processed_image}],
                            split_long_images,
                            {name: orig_b64_single},
                        )
                    )
                    
                except Exception as e:
                    print(f"Error processing {file.filename}: {e}")
                    continue
    
    if not processed_images:
        return redirect("/")
    
    return render_template("translate.html", images=processed_images)


@app.route("/prepare-ocr", methods=["POST"])
def prepare_ocr():
    """Detect + OCR only; hiển thị trang chỉnh text trước khi dịch."""
    cfg = parse_translate_form(request)
    if cfg is None:
        return redirect("/")

    gemini_api_key = cfg["gemini_api_key"]
    enable_black_bubble = cfg["enable_black_bubble"]
    selected_font = cfg["selected_font"]
    files = cfg["files"]

    if _OCR_CACHE["manga_ocr"] is None:
        _OCR_CACHE["manga_ocr"] = MangaOcr()
    mocr = _OCR_CACHE["manga_ocr"]

    font_analyzer = None
    if selected_font == "auto":
        try:
            from font_analyzer import FontAnalyzer

            api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Warning: No Gemini API key provided for font analysis")
            font_analyzer = FontAnalyzer(api_key=api_key)
        except Exception as e:
            print(f"Failed to initialize font analyzer: {e}")
            selected_font = "animeace_"

    all_images = []
    for file in files:
        if file and file.filename:
            try:
                file_stream = file.stream
                file_bytes = np.frombuffer(file_stream.read(), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    continue
                name = os.path.splitext(file.filename)[0]
                all_images.append({"image": image, "name": name})
            except Exception as e:
                print(f"Error reading {file.filename}: {e}")

    if not all_images:
        return redirect("/")

    if selected_font == "auto" and font_analyzer is not None:
        try:
            results = detect_bubbles(MODEL_PATH, all_images[0]["image"], enable_black_bubble)
            if results:
                det = results[0]
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                first_bubble = all_images[0]["image"][int(y1) : int(y2), int(x1) : int(x2)]
                selected_font = font_analyzer.analyze_and_match(first_bubble)
            else:
                selected_font = "animeace_"
        except Exception as e:
            print(f"Font analysis failed: {e}")
            selected_font = "animeace_"

    all_pages_data = detect_and_ocr_only(all_images, mocr, enable_black_bubble)

    st = cfg["selected_translator"]
    manga_translator = MangaTranslator(source=cfg["source_lang"], target=cfg["target_lang"])
    if st == "gemini" and cfg["style"]:
        manga_translator._gemini_custom_prompt = cfg["style"]
    if st == "copilot" and cfg["style"]:
        manga_translator._copilot_custom_prompt = cfg["style"]
    if st == "gemini" and cfg["gemini_api_key"]:
        manga_translator._gemini_api_key = cfg["gemini_api_key"]
    if st == "copilot":
        manga_translator._copilot_server = cfg["copilot_server"]
        manga_translator._copilot_model = cfg["copilot_model"]

    if st == "copilot":
        from translator.local_llm_translator import LocalLLMTranslator

        manga_translator._local_llm_translator = LocalLLMTranslator(
            server_url=cfg["copilot_server"],
            model=cfg["copilot_model"],
            custom_prompt=cfg["style"] if cfg["style"] else None,
        )
    elif st == "gemini" and cfg["gemini_api_key"]:
        from translator.gemini_translator import GeminiTranslator

        manga_translator._gemini_translator = GeminiTranslator(
            api_key=cfg["gemini_api_key"],
            custom_prompt=cfg["style"] if cfg["style"] else None,
        )

    _emit_progress_ws("translation", 0, 1, "Đang tạo bản dịch demo...")
    try:
        demo_by_page = translate_all_pages_preview(
            all_pages_data,
            manga_translator,
            st,
            batch_size=10,
            use_context_memory=cfg["use_context_memory"],
            emit_ws=False,
        )
    except Exception as e:
        print(f"Bản dịch demo thất bại: {e}")
        demo_by_page = {n: [] for n in all_pages_data}
    _emit_progress_ws("translation", 1, 1, "Bản dịch demo xong")

    def _align_demo_lines(ocr_list, demo_list):
        demo_list = demo_list or []
        return [demo_list[i] if i < len(demo_list) else "" for i in range(len(ocr_list))]

    job_id = str(uuid.uuid4())
    _prune_prepare_jobs()
    PREPARE_CACHE[job_id] = {
        "ts": time_module.time(),
        "all_pages_data": all_pages_data,
        "split_long_images": cfg["split_long_images"],
        "selected_translator": cfg["selected_translator"],
        "selected_font": selected_font,
        "use_context_memory": cfg["use_context_memory"],
        "enable_black_bubble": enable_black_bubble,
        "source_lang": cfg["source_lang"],
        "target_lang": cfg["target_lang"],
        "style": cfg["style"],
        "gemini_api_key": cfg["gemini_api_key"],
        "copilot_server": cfg["copilot_server"],
        "copilot_model": cfg["copilot_model"],
        "batch_size": 10,
    }

    pages = []
    for pi, (name, data) in enumerate(all_pages_data.items()):
        preview_src = data.get("image_original")
        if preview_src is None:
            preview_src = data["image"]
        _, buf = cv2.imencode(".jpg", preview_src, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY_PREVIEW])
        ocr_list = list(data["texts"])
        bubbles = data.get("bubbles", [])
        bubbles_json = []
        for b in bubbles:
            coords = b.get("coords", [0, 0, 0, 0])
            bubbles_json.append({
                "x": coords[0],
                "y": coords[1],
                "width": coords[2] - coords[0],
                "height": coords[3] - coords[1]
            })
        pages.append(
            {
                "index": pi,
                "name": name,
                "image_b64": base64.b64encode(buf.tobytes()).decode("utf-8"),
                "texts": ocr_list,
                "demo_texts": _align_demo_lines(ocr_list, demo_by_page.get(name, [])),
                "bubble_count": len(ocr_list),
                "bubbles_json": bubbles_json,
            }
        )

    return render_template("ocr_review.html", job_id=job_id, pages=pages)


@app.route("/complete-translation", methods=["POST"])
def complete_translation():
    """Áp dụng OCR đã chỉnh, dịch và render."""
    job_id = request.form.get("job_id", "").strip()
    _prune_prepare_jobs()
    job = PREPARE_CACHE.pop(job_id, None)
    if not job:
        return redirect("/")

    all_pages_data = job["all_pages_data"]
    apply_ocr_edits_from_form(request, all_pages_data)
    originals_b64 = snapshot_originals_jpeg_b64_from_pages(all_pages_data)

    manga_translator = MangaTranslator(source=job["source_lang"], target=job["target_lang"])
    st = job["selected_translator"]
    if st == "gemini" and job["style"]:
        manga_translator._gemini_custom_prompt = job["style"]
    if st == "copilot" and job["style"]:
        manga_translator._copilot_custom_prompt = job["style"]
    if st == "gemini" and job["gemini_api_key"]:
        manga_translator._gemini_api_key = job["gemini_api_key"]
    if st == "copilot":
        manga_translator._copilot_server = job["copilot_server"]
        manga_translator._copilot_model = job["copilot_model"]

    if st == "copilot":
        from translator.local_llm_translator import LocalLLMTranslator

        manga_translator._local_llm_translator = LocalLLMTranslator(
            server_url=job["copilot_server"],
            model=job["copilot_model"],
            custom_prompt=job["style"] if job["style"] else None,
        )
    elif st == "gemini":
        from translator.gemini_translator import GeminiTranslator

        if not job.get("gemini_api_key"):
            return redirect("/")
        manga_translator._gemini_translator = GeminiTranslator(
            api_key=job["gemini_api_key"],
            custom_prompt=job["style"] if job["style"] else None,
        )

    n_pages = len(all_pages_data)
    if st in ("gemini", "copilot"):
        processed_results = translation_and_render_gemini_copilot(
            all_pages_data,
            manga_translator,
            job["selected_font"],
            st,
            batch_size=job.get("batch_size", 10),
            use_context_memory=job["use_context_memory"],
            total_images=n_pages,
        )
    else:
        processed_results = translation_and_render_other_translators(
            all_pages_data,
            manga_translator,
            st,
            job["selected_font"],
            total_images=n_pages,
        )

    processed_images = encode_gallery_items(
        processed_results, job["split_long_images"], originals_b64
    )
    if not processed_images:
        return redirect("/")
    return render_template("translate.html", images=processed_images)


@app.route("/download-zip", methods=["POST"])
def download_zip():
    """Create and download a ZIP file containing all translated images."""
    try:
        images_data = request.form.get("images_data", "[]")
        images = json.loads(images_data)
        
        if not images:
            return redirect("/")
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for i, img in enumerate(images):
                name = img.get('name', f'image_{i+1}')
                data = img.get('data', '')
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(data)
                
                # Add to ZIP with proper filename
                filename = f"{name}_translated.png"
                zip_file.writestr(filename, image_bytes)
        
        zip_buffer.seek(0)
        
        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='manga_translated.zip'
        )
    
    except Exception as e:
        print(f"Error creating ZIP: {e}")
        return redirect("/")


if __name__ == "__main__":
    socketio.run(app, debug=True)
