import gradio as gr
import time
import torch
from typing import Dict, Any, List
import json
from models.model_registry import OptimizedModelRegistry
import yaml
import logging
from PIL import Image, ImageDraw, ImageFont

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedMultiTaskApp:
    def __init__(self, config_path: str = "configs/app_config.yaml"):
        self.config = self._load_config(config_path)
        self.model_registry = OptimizedModelRegistry(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _format_latency(self, seconds: float) -> str:
        """Format latency in appropriate units"""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        else:
            return f"{seconds:.1f}s"
    
    def _create_error_image(self, message: str):
        """Create an error image when generation fails"""
        img = Image.new('RGB', (300, 200), color='red')
        d = ImageDraw.Draw(img)
        d.text((10, 10), message, fill='white')
        return img
    
    def sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform sentiment analysis on input text"""
        start_time = time.time()
        
        try:
            sentiment_pipeline = self.model_registry.load_sentiment_analysis()
            if not sentiment_pipeline:
                return {"error": "Failed to load sentiment analysis model"}
            
            results = sentiment_pipeline(text[:self.config['limits']['max_text_length']])
            latency = time.time() - start_time
            
            return {
                "sentiment": results[0]['label'],
                "confidence": f"{results[0]['score']:.3f}",
                "latency": self._format_latency(latency),
                "text_length": len(text)
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {"error": str(e)}
    
    def text_summarization(self, text: str, max_length: int = 100) -> Dict[str, Any]:
        """Summarize input text"""
        start_time = time.time()
        
        try:
            summarization_pipeline = self.model_registry.load_summarization()
            if not summarization_pipeline:
                return {"error": "Failed to load summarization model"}
            
            max_len = min(max_length, self.config['limits']['max_summary_length'])
            summary = summarization_pipeline(
                text[:self.config['limits']['max_text_length']],
                max_length=max_len,
                min_length=30,
                do_sample=False
            )
            
            latency = time.time() - start_time
            compression_ratio = len(summary[0]['summary_text']) / len(text) if len(text) > 0 else 0
            
            return {
                "summary": summary[0]['summary_text'],
                "original_length": len(text),
                "summary_length": len(summary[0]['summary_text']),
                "compression_ratio": f"{compression_ratio:.2f}",
                "latency": self._format_latency(latency)
            }
            
        except Exception as e:
            logger.error(f"Error in text summarization: {e}")
            return {"error": str(e)}
    
    def next_word_prediction(self, text: str, num_predictions: int = 5) -> Dict[str, Any]:
        """Predict next words"""
        start_time = time.time()
        
        try:
            model, tokenizer = self.model_registry.load_next_word_prediction()
            if not model or not tokenizer:
                return {"error": "Failed to load next-word prediction model"}
            
            inputs = tokenizer.encode(text, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(inputs)
                next_token_logits = outputs.logits[:, -1, :]
                probabilities = torch.softmax(next_token_logits, dim=-1)
                
                top_probs, top_indices = torch.topk(probabilities, num_predictions, dim=-1)
                
                predictions = []
                for i in range(num_predictions):
                    token_id = top_indices[0][i].item()
                    word = tokenizer.decode([token_id])
                    prob = top_probs[0][i].item()
                    predictions.append({
                        "word": word.strip(),
                        "probability": f"{prob:.4f}"
                    })
            
            latency = time.time() - start_time
            
            return {
                "predictions": predictions,
                "input_text": text,
                "latency": self._format_latency(latency)
            }
            
        except Exception as e:
            logger.error(f"Error in next word prediction: {e}")
            return {"error": str(e)}
    
    def text_translation(self, text: str) -> Dict[str, Any]:
        """Translate text to French"""
        start_time = time.time()
        
        try:
            translation_pipeline = self.model_registry.load_translation()
            if not translation_pipeline:
                return {"error": "Failed to load translation model"}
            
            translation = translation_pipeline(text[:self.config['limits']['max_text_length']])
            latency = time.time() - start_time
            
            return {
                "translated_text": translation[0]['translation_text'],
                "original_text": text,
                "target_language": "French",
                "latency": self._format_latency(latency)
            }
            
        except Exception as e:
            logger.error(f"Error in text translation: {e}")
            return {"error": str(e)}
    
    def named_entity_recognition(self, text: str) -> Dict[str, Any]:
        """Perform named entity recognition"""
        start_time = time.time()
        
        try:
            ner_pipeline = self.model_registry.load_ner()
            if not ner_pipeline:
                return {"error": "Failed to load NER model"}
            
            entities = ner_pipeline(text[:self.config['limits']['max_text_length']])
            
            formatted_entities = []
            for entity in entities:
                formatted_entities.append({
                    "entity": entity['entity_group'],
                    "word": entity['word'],
                    "score": f"{entity['score']:.3f}",
                    "start": entity['start'],
                    "end": entity['end']
                })
            
            latency = time.time() - start_time
            
            return {
                "entities": formatted_entities,
                "total_entities": len(formatted_entities),
                "latency": self._format_latency(latency)
            }
            
        except Exception as e:
            logger.error(f"Error in named entity recognition: {e}")
            return {"error": str(e)}
    
    def image_generation(self, prompt: str, progress=gr.Progress()):
        """Generate image from text prompt with progress tracking"""
        start_time = time.time()
        
        try:
            pipe = self.model_registry.load_image_generation()
            if not pipe:
                error_msg = "Failed to load image generation model"
                logger.error(error_msg)
                error_image = self._create_error_image("Model Loading Failed")
                return error_image, {
                    "error": error_msg,
                    "prompt": prompt,
                    "status": "❌ Model loading failed"
                }
            
            num_steps = self.config['models']['image_generation']['num_inference_steps']
            guidance_scale = self.config['models']['image_generation']['guidance_scale']
            
            def callback(step, timestep, latents):
                if progress:
                    progress((step + 1) / num_steps, f"Generating image... Step {step+1}/{num_steps}")
            
            logger.info(f"🔄 Generating image with {num_steps} steps...")
            
            with torch.no_grad():
                image = pipe(
                    prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    callback=callback if progress else None
                ).images[0]
            
            latency = time.time() - start_time
            
            return image, {
                "prompt": prompt,
                "inference_steps": num_steps,
                "latency": self._format_latency(latency),
                "status": "✅ Generation completed successfully!"
            }
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            logger.error(f"Error in image generation: {e}")
            error_image = self._create_error_image("Generation Error")
            return error_image, {
                "error": error_msg,
                "prompt": prompt,
                "status": "❌ Generation failed"
            }
    
    def create_interface(self):
        """Create optimized Gradio interface"""
        with gr.Blocks(
            title=self.config['app']['title'],
            theme=self.config['app']['theme']
        ) as demo:
            gr.Markdown(f"""
            # 🚀 {self.config['app']['title']}
            **Multi-Task AI Application - Optimized for CPU Performance**
            """)
            
            with gr.Tabs():
                # Sentiment Analysis Tab
                with gr.TabItem("😊 Sentiment Analysis"):
                    with gr.Row():
                        with gr.Column():
                            sentiment_input = gr.Textbox(
                                label="Input Text",
                                placeholder="Enter text for sentiment analysis...",
                                lines=3
                            )
                            sentiment_btn = gr.Button("Analyze Sentiment", variant="primary")
                        with gr.Column():
                            sentiment_output = gr.JSON(label="Analysis Results")
                    
                    examples = [
                        "I absolutely love this product! It's amazing!",
                        "This is the worst experience I've ever had.",
                        "The weather is nice today."
                    ]
                    gr.Examples(examples=examples, inputs=sentiment_input)
                
                # Text Summarization Tab
                with gr.TabItem("📝 Text Summarization"):
                    with gr.Row():
                        with gr.Column():
                            summary_input = gr.Textbox(
                                label="Input Text",
                                placeholder="Enter long text to summarize...",
                                lines=5
                            )
                            summary_length = gr.Slider(
                                minimum=30, maximum=150, value=100, step=10,
                                label="Summary Length"
                            )
                            summary_btn = gr.Button("Generate Summary", variant="primary")
                        with gr.Column():
                            summary_output = gr.JSON(label="Summary Results")
                    
                    examples = [
                        "The quick brown fox jumps over the lazy dog. This is a classic sentence used in typing tests and font demonstrations. It contains all the letters of the English alphabet, making it a perfect pangram. Pangrams are useful for displaying typefaces and testing equipment."
                    ]
                    gr.Examples(examples=examples, inputs=summary_input)
                
                # Next Word Prediction Tab
                with gr.TabItem("🔮 Next Word Prediction"):
                    with gr.Row():
                        with gr.Column():
                            next_word_input = gr.Textbox(
                                label="Input Text",
                                placeholder="Start typing...",
                                lines=2
                            )
                            num_predictions = gr.Slider(
                                minimum=1, maximum=10, value=5, step=1,
                                label="Number of Predictions"
                            )
                            next_word_btn = gr.Button("Predict Next Words", variant="primary")
                        with gr.Column():
                            next_word_output = gr.JSON(label="Prediction Results")
                    
                    examples = [
                        "The weather today is",
                        "I want to eat some",
                        "Machine learning is"
                    ]
                    gr.Examples(examples=examples, inputs=next_word_input)
                
                # Translation Tab
                with gr.TabItem("🌐 Translation"):
                    with gr.Row():
                        with gr.Column():
                            translation_input = gr.Textbox(
                                label="Input Text (English)",
                                placeholder="Enter English text to translate...",
                                lines=3
                            )
                            translation_btn = gr.Button("Translate to French", variant="primary")
                        with gr.Column():
                            translation_output = gr.JSON(label="Translation Results")
                    
                    examples = [
                        "Hello, how are you?",
                        "This is a beautiful day.",
                        "I love machine learning."
                    ]
                    gr.Examples(examples=examples, inputs=translation_input)
                
                # Named Entity Recognition Tab
                with gr.TabItem("🏷️ Named Entity Recognition"):
                    with gr.Row():
                        with gr.Column():
                            ner_input = gr.Textbox(
                                label="Input Text",
                                placeholder="Enter text to extract entities...",
                                lines=3
                            )
                            ner_btn = gr.Button("Extract Entities", variant="primary")
                        with gr.Column():
                            ner_output = gr.JSON(label="Entity Results")
                    
                    examples = [
                        "John Smith works at Google in California.",
                        "Apple Inc. is located in Cupertino, California.",
                        "Marie Curie won the Nobel Prize in Paris."
                    ]
                    gr.Examples(examples=examples, inputs=ner_input)
                
                # Image Generation Tab
                with gr.TabItem("🎨 Image Generation"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### ⚡ CPU-Optimized Image Generation")
                            image_input = gr.Textbox(
                                label="Prompt",
                                placeholder="Describe the image you want to generate...",
                                lines=2,
                                value="A beautiful sunset over mountains, digital art"
                            )
                            image_btn = gr.Button("🚀 Generate Image", variant="primary")
                            clear_btn = gr.Button("Clear", variant="secondary")
                        with gr.Column():
                            image_output = gr.Image(label="Generated Image", height=300)
                            image_info = gr.JSON(label="Generation Info")
                    
                    examples = [
                        "A beautiful sunset over mountains, digital art",
                        "A cat sitting on a bookshelf, cartoon style",
                        "A futuristic city with flying cars, sci-fi art"
                    ]
                    gr.Examples(examples=examples, inputs=image_input)
            
            # Connect buttons to functions
            sentiment_btn.click(
                fn=self.sentiment_analysis,
                inputs=[sentiment_input],
                outputs=[sentiment_output]
            )
            
            summary_btn.click(
                fn=self.text_summarization,
                inputs=[summary_input, summary_length],
                outputs=[summary_output]
            )
            
            next_word_btn.click(
                fn=self.next_word_prediction,
                inputs=[next_word_input, num_predictions],
                outputs=[next_word_output]
            )
            
            translation_btn.click(
                fn=self.text_translation,
                inputs=[translation_input],
                outputs=[translation_output]
            )
            
            ner_btn.click(
                fn=self.named_entity_recognition,
                inputs=[ner_input],
                outputs=[ner_output]
            )
            
            def clear_image():
                return None, {}
            
            image_btn.click(
                fn=self.image_generation,
                inputs=[image_input],
                outputs=[image_output, image_info]
            )
            
            clear_btn.click(
                fn=clear_image,
                inputs=[],
                outputs=[image_output, image_info]
            )
        
        return demo

def main():
    """Main function to run the optimized application"""
    # Set CPU-only mode
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    print("🚀 Starting Multi-Task Gradio App...")
    print("📝 This may take a few minutes to load all models...")
    print("🌐 App will be available at: http://localhost:7860")
    print("⚡ Running in CPU-Only mode")
    
    app = OptimizedMultiTaskApp()
    demo = app.create_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main()