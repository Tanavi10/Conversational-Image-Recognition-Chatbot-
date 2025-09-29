import gradio as gr
import os
import socket
from contextlib import closing
from utils.yolo_detection import YOLOv12Detector
from utils.gemini_captioning import GeminiCaptioner
from utils.gemini_vqa import GeminiVQA
import time
import numpy as np

# Load API key
with open("utils/gemini_api_key.txt") as f:
    GEMINI_API_KEY = f.read().strip()

# Custom CSS - Updated to hide image icons
contrast_css = """
.gr-image-preview .icon {
    display: none !important;
}

.gr-image-preview {
    background-color: transparent !important;
    border: none !important;
}

/* Additional styling from before preserved */
:root {
    --primary: #6e48aa;
    --secondary: #9d50bb;
    --accent: #4776E6;
    --dark: #1a1a2e;
    --light: #f8f9fa;
}

.gradio-container {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, var(--dark) 0%, #16213E 100%);
    color: var(--light);
    font-size: 18px;
}

.header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    border-radius: 0 0 20px 20px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}

.header h1 {
    color: white;
    font-size: 3.2rem;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    font-weight: bold;
}

.header p {
    color: rgba(255,255,255,0.9);
    font-size: 1.5rem;
}

.image-box {
    border-radius: 15px;
    border: 3px solid var(--accent);
    box-shadow: 0 15px 30px rgba(0,0,0,0.2);
    background: var(--dark);
    transition: all 0.3s ease;
}

.image-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
}

.output-box {
    border-radius: 15px;
    background: rgba(30,30,46,0.8);
    padding: 1.5rem;
    margin-top: 1rem;
    border-left: 5px solid var(--accent);
    font-size: 1.2rem;
}
"""

def find_free_port(start_port=7860, end_port=8000):
    for port in range(start_port, end_port + 1):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(('', port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
            except OSError:
                continue
    raise OSError(f"No free ports available between {start_port}-{end_port}")

captioner = None
vqa = None

def initialize_models():
    try:
        print("🔄 Loading models...")
        captioner = GeminiCaptioner(GEMINI_API_KEY)
        vqa = GeminiVQA(GEMINI_API_KEY)
        detector = YOLOv12Detector()
        print("✅ Models loaded successfully")
        return captioner, detector, vqa
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        raise

captioner, detector, vqa = initialize_models()

def process_image(image):
    if image is None:
        return None, "🛑 Please upload an image first", None
    try:
        import cv2
        os.makedirs("temp", exist_ok=True)
        image_path = "temp/uploaded.jpg"

        if isinstance(image, np.ndarray):
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            image.save(image_path)

        yield None, "🔍 Analyzing image...", None

        detected_image = detector.detect(image_path)
        yield detected_image, "📝 Generating caption...", image_path

        caption = captioner.caption(image_path)
        yield detected_image, f"✨ {caption}", image_path

    except Exception as e:
        yield None, f"⚠ Error: {str(e)}", None

def process_question(image_path, question):
    if not image_path or not question.strip():
        return "🛑 Missing image or question"
    try:
        for i in range(3):
            yield f"💭 Thinking{'...'[:i+1]}"
            time.sleep(0.5)

        answer = vqa.answer(image_path, question)
        yield f"💡 {answer}" if answer else "🤷 No answer generated"
    except Exception as e:
        yield f"⚠ Error: {str(e)}"

with gr.Blocks(title="Vision AI Chatbot", css=contrast_css) as demo:
    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class="header">
                <h1>Vision AI Chatbot</h1>
                <p>Upload an image, ask questions, and get AI-powered visual analysis</p>
            </div>
            """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(show_label=False, show_download_button=False, elem_classes=["image-box"])
            analyze_btn = gr.Button("Analyze Image")

        with gr.Column(scale=1):
            output_image = gr.Image(show_label=False, show_download_button=False, elem_classes=["image-box"])
            output_text = gr.Markdown(elem_classes=["output-box"])
            image_path_state = gr.State(None)

    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Ask a question about the image",
                placeholder="What objects are in the image?",
                elem_classes=["textbox"]
            )
            ask_btn = gr.Button("Ask Question")
            answer_output = gr.Markdown(elem_classes=["output-box"])

    with gr.Row():
        with gr.Column():
            gr.HTML("""
            <div class="footer">
                <p>© 2025 TanJutBot | Powered by YOLOv12, Gemini 1.5</p>
            </div>
            """)

    analyze_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[output_image, output_text, image_path_state],
    )

    ask_btn.click(
        fn=process_question,
        inputs=[image_path_state, question_input],
        outputs=[answer_output],
    )

if __name__ == "__main__":
    try:
        port = find_free_port()
        print(f"🚀 Launching on port: {port}")
        print(f"👉 Please open: http://localhost:{port} in your browser")
        demo.launch(
            server_port=port,
            share=False,
            server_name="0.0.0.0",
            show_error=True,
            debug=True
        )
    except Exception as e:
        print(f"❌ Launch failed: {e}")
    demo.launch()
