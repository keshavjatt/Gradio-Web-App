# Updated app - Rust-free version
import gradio as gr
import torch
from model_loader import model_loader
import time

print("Loading simplified models...")
print(model_loader.load_sentiment_model())
print(model_loader.load_summarization_model())
print(model_loader.load_text_generation_model())
print(model_loader.load_image_generation_model())
print("All models loaded! Starting app...")

# 1. SENTIMENT ANALYSIS (Updated)
def analyze_sentiment(text):
    if not text.strip():
        return "❌ Please enter some text"
    
    try:
        start_time = time.time()
        result = model_loader.models['sentiment'](text)[0]
        time_taken = round(time.time() - start_time, 2)
        
        label = result['label']
        score = round(result['score'] * 100, 2)
        
        return f"Emotion: {label}\nConfidence: {score}%\nTime: {time_taken}s"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 2. TEXT SUMMARIZATION (Updated)
def summarize_text(text):
    if not text.strip():
        return "❌ Please enter some text"
    
    try:
        start_time = time.time()
        
        summary = model_loader.models['summarization'](
            text,
            max_length=100,
            min_length=30,
            do_sample=False
        )[0]['summary_text']
        
        time_taken = round(time.time() - start_time, 2)
        
        return f"📝 Summary:\n{summary}\n\n⏰ Time: {time_taken}s"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 3. TEXT GENERATION (Updated - pipeline use karo)
def predict_next_words(text, num_words=20):
    if not text.strip():
        return "❌ Please enter some text"
    
    try:
        start_time = time.time()
        
        # Direct pipeline use karo
        generated = model_loader.models['text_generation'](
            text,
            max_new_tokens=num_words,
            temperature=0.7,
            do_sample=True,
            pad_token_id=50256  # GPT2 ka eos token
        )[0]['generated_text']
        
        time_taken = round(time.time() - start_time, 2)
        
        return f"📖 Generated Text:\n{generated}\n\n⏰ Time: {time_taken}s"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

# 4. IMAGE GENERATION (Same)
def generate_image(prompt):
    if not prompt.strip():
        return None, "❌ Please enter a prompt"
    
    try:
        start_time = time.time()
        
        inappropriate_words = ['nude', 'naked', 'violence', 'blood', 'kill']
        if any(word in prompt.lower() for word in inappropriate_words):
            return None, "❌ Inappropriate content detected."
        
        with torch.no_grad():
            image = model_loader.models['image_generation'](
                prompt,
                num_inference_steps=15,  # Even faster
                guidance_scale=7.5
            ).images[0]
        
        time_taken = round(time.time() - start_time, 2)
        
        return image, f"✅ Image generated!\n⏰ Time: {time_taken}s"
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# GRADIO INTERFACE (Same structure)
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="Simple AI App") as app:
        gr.Markdown("# 🚀 Simple Multi-Task AI App")
        
        with gr.Tab("😊 Sentiment Analysis"):
            gr.Markdown("### Analyze Text Emotion")
            with gr.Row():
                with gr.Column():
                    sentiment_input = gr.Textbox(
                        label="Enter text",
                        placeholder="I'm feeling happy today...",
                        lines=3
                    )
                    sentiment_button = gr.Button("Analyze")
                with gr.Column():
                    sentiment_output = gr.Textbox(label="Result", lines=4)
            
            sentiment_button.click(analyze_sentiment, inputs=sentiment_input, outputs=sentiment_output)
        
        with gr.Tab("📝 Text Summarization"):
            gr.Markdown("### Summarize Text")
            with gr.Row():
                with gr.Column():
                    summary_input = gr.Textbox(
                        label="Enter long text",
                        placeholder="Paste your text here...",
                        lines=6
                    )
                    summary_button = gr.Button("Summarize")
                with gr.Column():
                    summary_output = gr.Textbox(label="Summary", lines=5)
            
            summary_button.click(summarize_text, inputs=summary_input, outputs=summary_output)
        
        with gr.Tab("🔮 Text Generation"):
            gr.Markdown("### Generate Text")
            with gr.Row():
                with gr.Column():
                    textgen_input = gr.Textbox(
                        label="Start writing",
                        placeholder="The future of AI is...",
                        lines=3
                    )
                    textgen_words = gr.Slider(5, 40, value=20, label="Words to generate")
                    textgen_button = gr.Button("Generate")
                with gr.Column():
                    textgen_output = gr.Textbox(label="Result", lines=5)
            
            textgen_button.click(predict_next_words, inputs=[textgen_input, textgen_words], outputs=textgen_output)
        
        with gr.Tab("🎨 Image Generation"):
            gr.Markdown("### Generate Image")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Textbox(
                        label="Describe your image",
                        placeholder="A cute cat reading a book",
                        lines=2
                    )
                    image_button = gr.Button("Generate Image")
                    image_status = gr.Textbox(label="Status", lines=2)
                with gr.Column():
                    image_output = gr.Image(label="Generated Image", height=300)
            
            image_button.click(generate_image, inputs=image_input, outputs=[image_output, image_status])
    
    return app

if __name__ == "__main__":
    print("🚀 Starting Simple AI App...")
    print("📧 Open: http://localhost:7860")
    print("⏹️  Press Ctrl+C to stop")
    
    app = create_interface()
    app.launch(server_name="0.0.0.0", share=False, inbrowser=True)