import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Configuration
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizer warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only mode

class LiteImageCaptioner:
    """
    Memory-efficient image captioning using BLIP (not BLIP-2) with:
    - CPU-only operation
    - Lower memory footprint
    - Basic error handling
    - Support for local files and PIL Images
    """
    
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        """Initialize with smaller BLIP model"""
        self.device = "cpu"
        try:
            print("🔄 Loading lightweight captioning model...")
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32  # Using float32 for CPU stability
            ).to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def generate_caption(
        self, 
        image_input,
        prompt: str = None,
        max_length: int = 50,  # Reduced from 100
        temperature: float = 0.7
    ) -> str:
        """
        Generate caption for an image
        
        Args:
            image_input: Path or PIL Image
            prompt: Optional text prompt
            max_length: Shorter default max length
            temperature: Creativity control
            
        Returns:
            Caption string or error message
        """
        try:
            # Load image
            image = self._load_image(image_input)
            if image is None:
                return "⚠️ Could not load image"
            
            # Reduce image size if too large
            if max(image.size) > 512:
                image = self._resize_image(image)
            
            # Prepare inputs with smaller batch size
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate with constrained resources
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    num_beams=3,  # Fewer beams for faster generation
                    early_stopping=True
                )
            
            return self.processor.decode(outputs[0], skip_special_tokens=True).strip()
            
        except Exception as e:
            print(f"⚠️ Captioning error: {e}")
            return f"Error generating caption"

    def _load_image(self, image_input):
        """Handle image loading with basic URL support removed"""
        if isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        
        if isinstance(image_input, str):
            try:
                return Image.open(image_input).convert('RGB')
            except:
                return None
        return None

    def _resize_image(self, image, max_size=512):
        """Resize large images to save memory"""
        width, height = image.size
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return image.resize((new_width, new_height))

# Example usage
if __name__ == "__main__":
    try:
        print("🚀 Starting lightweight captioning system...")
        captioner = LiteImageCaptioner()
        
        # Test with local image
        test_image = "examples/market_place.jpeg"
        if os.path.exists(test_image):
            print(f"\n📷 Processing: {test_image}")
            caption = captioner.generate_caption(
                test_image,
                prompt="a photograph of",
                max_length=30  # Even shorter for testing
            )
            print(f"💬 Caption: {caption}")
        else:
            print("⚠️ Test image not found")
            
        # Test with PIL Image
        pil_image = Image.new('RGB', (300, 200), color='red')
        print("\n📷 Processing PIL Image...")
        caption = captioner.generate_caption(pil_image)
        print(f"💬 Caption: {caption}")
        
    except Exception as e:
        print(f"❌ System failed: {e}")