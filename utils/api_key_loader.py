# utils/api_key_loader.py

def load_gemini_api_key(path="utils/gemini_api_key.txt"):
    try:
        with open(path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        raise RuntimeError("Gemini API key file not found. Please check the path.")
