# test_caption.py

import os
from utils.gemini_captioning import GeminiCaptioner  

if __name__ == "_main__":
    api_key_path = "utils/gemini_api_key.txt"
    image_path = "examples/market_place.jpeg"

    if os.path.exists(api_key_path) and os.path.exists(image_path):
        with open(api_key_path) as f:
            api_key = f.read().strip()
        captioner = GeminiCaptioner(api_key)
        print("🔍 Generating caption...")
        print("💬 Caption:", captioner.caption(image_path))
    else:
        print("❌ API key or image not found")