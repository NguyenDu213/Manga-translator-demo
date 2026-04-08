import cv2
import numpy as np


def get_bubble_background_color(image, mask=None):
    """
    Lấy màu nền đặc trưng của vùng bubble (không lấy pixel chữ).
    Nếu có mask → chỉ lấy vùng mask (vùng đã xác định là interior).
    Nếu không → lấy phần lớn nhất không phải chữ bằng k-means.
    """
    if mask is not None:
        pixels = image[mask == 255]
    else:
        pixels = image.reshape(-1, 3)

    if len(pixels) == 0:
        return (255, 255, 255)

    pixels = np.float32(pixels)

    # K-means: tìm 3 màu chính, lấy màu chiếm nhiều nhất (nền bubble)
    # Loại bỏ màu chữ (thường rất đen hoặc rất trắng)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = min(3, len(pixels))
    if k < 1:
        return (255, 255, 255)

    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    unique, counts = np.unique(labels, return_counts=True)
    dominant_idx = unique[np.argmax(counts)]
    dominant_color = centers[dominant_idx]

    return tuple(int(c) for c in dominant_color)


def get_color_by_histogram(pixels, bins=32):
    """
    Find dominant color using histogram binning.
    More accurate than simple mean/median for multimodal distributions.
    
    Args:
        pixels: Array of pixel colors (N, 3)
        bins: Number of bins per channel
        
    Returns:
        tuple: Dominant color as (B, G, R)
    """
    if len(pixels) == 0:
        return (255, 255, 255)
    
    # Create 3D histogram
    hist_b = np.histogram(pixels[:, 0], bins=bins, range=(0, 256))[0]
    hist_g = np.histogram(pixels[:, 1], bins=bins, range=(0, 256))[0]
    hist_r = np.histogram(pixels[:, 2], bins=bins, range=(0, 256))[0]
    
    # Find peak bin for each channel
    bin_width = 256 // bins
    b_peak = np.argmax(hist_b) * bin_width + bin_width // 2
    g_peak = np.argmax(hist_g) * bin_width + bin_width // 2
    r_peak = np.argmax(hist_r) * bin_width + bin_width // 2
    
    return (int(b_peak), int(g_peak), int(r_peak))


def get_color_by_mode(pixels):
    """
    Find the most frequent color (mode) in pixel array.
    Uses color quantization to group similar colors.
    
    Args:
        pixels: Array of pixel colors (N, 3)
        
    Returns:
        tuple: Most frequent color as (B, G, R)
    """
    if len(pixels) == 0:
        return (255, 255, 255)
    
    # Quantize colors (reduce to 32 levels per channel)
    quantized = (pixels // 8) * 8
    
    # Convert to hashable format for counting
    color_codes = quantized[:, 0] * 65536 + quantized[:, 1] * 256 + quantized[:, 2]
    
    # Find most frequent
    unique, counts = np.unique(color_codes, return_counts=True)
    most_freq_code = unique[np.argmax(counts)]
    
    # Decode back to BGR
    b = (most_freq_code // 65536) % 256
    g = (most_freq_code // 256) % 256
    r = most_freq_code % 256
    
    return (int(b), int(g), int(r))


def get_bubble_background_color(image, mask=None):
    """
    Detect bubble background color using histogram quantization.
    ~10x faster than k-means; works by quantizing pixels to 32 levels
    and finding the most frequent color cluster.
    
    Args:
        image: Input bubble image (BGR)
        mask: Optional interior mask. If provided, only masked pixels are sampled.
              
    Returns: 
        tuple: Background color as (B, G, R)
    """
    h, w = image.shape[:2]

    if mask is not None:
        pixels = image[mask == 255]
        if len(pixels) < 50:
            pixels = image.reshape(-1, 3)
    else:
        # Sample center region to avoid edge artifacts from oversize bbox
        cx, cy = w // 6, h // 6
        center = image[cy:h - cy, cx:w - cx]
        pixels = center.reshape(-1, 3) if center.size > 0 else image.reshape(-1, 3)

    if len(pixels) < 10:
        return (255, 255, 255)

    # Quantize to 32 levels per channel (~4x speedup over k-means)
    quantized = (pixels.astype(np.uint8) // 8) * 8
    codes = (
        quantized[:, 0].astype(np.uint32) * 4096
        + quantized[:, 1].astype(np.uint32) * 64
        + quantized[:, 2].astype(np.uint32)
    )
    unique, counts = np.unique(codes, return_counts=True)
    most_freq_idx = np.argmax(counts)
    b = (unique[most_freq_idx] // 4096) % 256
    g = (unique[most_freq_idx] // 64) % 256
    r = unique[most_freq_idx] % 256
    return (int(b), int(g), int(r))


def is_dark_bubble(image, threshold=100):
    """
    Determine if a bubble image is dark (black bubble with white text).
    
    Args:
        image: Input bubble image (BGR)
        threshold: Intensity threshold (below = dark bubble)
        
    Returns:
        bool: True if dark bubble, False if light bubble
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    return mean_intensity < threshold


def process_dark_bubble(image, fill_color=None):
    """
    Processes a dark speech bubble (black with white text).
    Fills the bubble contents with the detected or specified color.
    
    Args:
        image (numpy.ndarray): Input dark bubble image.
        fill_color: Color to fill (None = auto-detect)
        
    Returns:
        tuple: (processed_image, largest_contour, fill_color_used)
    """
    if fill_color is None:
        fill_color = get_bubble_background_color(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # For dark bubbles, find the dark region (invert threshold)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        image[:] = fill_color
        return image, largest_contour, fill_color
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)
    
    bg_color = get_bubble_background_color(image, mask)
    image[mask == 255] = bg_color
    
    return image, largest_contour, bg_color


def process_bubble(image, fill_color=None):
    """
    Xử lý bubble sáng: threshold + fill nền, trả về contour + màu nền đúng.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    bg_intensity = np.mean(fill_color) if fill_color is not None else 200
    if bg_intensity > 200:
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    elif bg_intensity < 50:
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    else:
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
        image[:] = fill_color or (255, 255, 255)
        return image, largest_contour, fill_color or (255, 255, 255)

    largest_contour = max(contours, key=cv2.contourArea)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, 255, cv2.FILLED)

    bg_color = get_bubble_background_color(image, mask)
    image[mask == 255] = bg_color

    return image, largest_contour, bg_color


def process_bubble_auto(image, force_dark=False, custom_color=None):
    """
    Automatically detect bubble type and process accordingly.
    
    Args:
        image: Input bubble image (BGR)
        force_dark: If True, treat as dark bubble regardless of detection
        custom_color: Custom color (B, G, R) - None = auto-detect
        
    Returns:
        tuple: (processed_image, contour, is_dark, detected_color)
    """
    # Auto-detect background color if no custom_color provided
    if custom_color is None:
        detected_color = get_bubble_background_color(image)
    else:
        detected_color = custom_color
    
    if force_dark or is_dark_bubble(image):
        processed, contour, color_used = process_dark_bubble(image, detected_color)
        return processed, contour, True, color_used
    else:
        processed, contour, color_used = process_bubble(image, detected_color)
        return processed, contour, False, color_used


def process_bubble_preserve_gradient(image, text_mask=None):
    """
    Process speech bubble while preserving gradient/complex backgrounds.
    Only removes text, keeps original background using inpainting.
    
    Args:
        image: Input bubble image (BGR)
        text_mask: Mask of text region to remove (None = auto-detect)
        
    Returns:
        tuple: (processed_image, contour)
    """
    if text_mask is None:
        # Auto-detect text using edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect text based on contrast
        bg_color = get_bubble_background_color(image)
        bg_intensity = np.mean(bg_color)
        
        if bg_intensity > 128:
            # Light background, dark text
            _, text_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        else:
            # Dark background, light text
            _, text_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Use inpainting to remove text and preserve background
    # Dilate mask slightly to ensure complete text removal
    kernel = np.ones((3, 3), np.uint8)
    text_mask_dilated = cv2.dilate(text_mask, kernel, iterations=1)
    
    # Inpaint to fill text region with surrounding background
    result = cv2.inpaint(image, text_mask_dilated, 3, cv2.INPAINT_TELEA)
    
    # Find bubble contour
    contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        h, w = image.shape[:2]
        largest_contour = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
    
    return result, largest_contour
