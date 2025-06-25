
import google.generativeai as genai
import os
from utils.gemini_vqa import GeminiVQA 


if __name__ == "main":
    import os
    api_key_path = "utils/gemini_api_key.txt"
    image_path = "examples/market_place.jpeg"
    question = "What items are being sold in this image?"

    if os.path.exists(api_key_path) and os.path.exists(image_path):
        with open(api_key_path) as f:
            api_key = f.read().strip()
        vqa = GeminiVQA(api_key)
        print(vqa.answer(image_path, question))
    else:
        print("API key or image not found")