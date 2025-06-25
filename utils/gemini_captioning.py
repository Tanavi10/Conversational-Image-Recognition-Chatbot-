import google.generativeai as genai
from PIL import Image

class GeminiCaptioner:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def caption(self, image_path, prompt="Describe this image"):
        """
        Generate an image caption using Gemini Vision API.

        Args:
            image_path (str): Path to the image file.
            prompt (str): Optional prompt for better caption quality.

        Returns:
            str: Generated caption text.
        """
        try:
            image = Image.open(image_path).convert("RGB")  # ✅ PIL Image for Gemini
            response = self.model.generate_content([prompt, image])
            return response.text.strip()
        except Exception as e:
            return f"❌ Gemini Captioning Error: {e}"