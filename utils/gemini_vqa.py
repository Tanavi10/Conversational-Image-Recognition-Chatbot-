import google.generativeai as genai
from PIL import Image

class GeminiVQA:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def answer(self, image_path, question):
        """
        Answer a question about an image using Gemini Vision API.

        Args:
            image_path (str): Path to the uploaded image file.
            question (str): Question to answer.

        Returns:
            str: Answer text.
        """
        try:
            image = Image.open(image_path).convert("RGB")  # ✅ PIL Image
            response = self.model.generate_content([question, image])
            return response.text.strip()
        except Exception as e:
            return f"❌ Gemini VQA Error: {e}"