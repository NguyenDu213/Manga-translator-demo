# Manga Translator 📚

Dịch tự động speech bubbles trong manga/manhwa/manhua với AI!

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **YOLO Detection** | Phát hiện speech bubble tự động (kể cả bubble đen) |
| 📝 **OCR** | Manga-OCR (cục bộ) |
| 🌐 **Translators** | Gemini, Google Translate |
| 🧠 **Context Memory** | Dịch chính xác hơn với context từ nhiều trang |
| 🎨 **24+ Fonts** | Auto font matching với Gemini Vision |
| 📦 **Download ZIP** | Tải tất cả ảnh đã dịch |

## � Quick Start

```bash
# Clone
cd Manga-Translator

# Install
pip install -r requirements.txt

# Run
python app.py
```

Mở http://localhost:5000

## � Translators

### Gemini (Recommended)
- Lấy API key từ [aistudio.google.com](https://aistudio.google.com/)
- Free tier: 15 RPM, 1M tokens/day

### Google Translate
- Dịch qua `deep-translator` (cần mạng)

## �📋 Workflow

1. **Upload** manga/manhwa images
2. **Chọn ngôn ngữ** (Japanese/Chinese/Korean → Vietnamese/English/...)
3. **Chọn translator** (Gemini hoặc Google)
4. **Click Translate** và xem progress real-time (Context Memory + scan bubble đen bật mặc định trong backend)
5. **Download** từng ảnh hoặc ZIP

## 🌍 Languages

**Source:** Japanese, Chinese, Korean, English  
**Target:** Vietnamese, English, Chinese, Korean, Thai, Indonesian, French, German, Spanish, Russian

##  Tech Stack

- **Backend:** Flask + Flask-SocketIO
- **Detection:** YOLOv8 + OpenCV (black bubbles)
- **OCR:** Manga-OCR
- **Translation:** Gemini API, Google Translate (deep-translator)
- **Rendering:** PIL with smart text wrapping

##  License

MIT
