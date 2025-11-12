# Yeh simplified model loader hai - koi Rust dependency nahi
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import warnings
warnings.filterwarnings("ignore")

class SimpleModelLoader:
    def __init__(self):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
    
    def load_sentiment_model(self):
        """Simple sentiment analysis model"""
        try:
            # Ye model Rust ki zaroorat nahi dalta
            sentiment_pipeline = pipeline(
                "text-classification",
                model="bhadresh-savani/distilbert-base-uncased-emotion",
                device=self.device
            )
            self.models['sentiment'] = sentiment_pipeline
            return "✅ Sentiment model loaded successfully!"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def load_summarization_model(self):
        """Simple summarization model"""
        try:
            # Alternative model jo Rust-free hai
            summarization_pipeline = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",
                device=self.device
            )
            self.models['summarization'] = summarization_pipeline
            return "✅ Summarization model loaded successfully!"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def load_text_generation_model(self):
        """Simple text generation - pipeline use karo"""
        try:
            # Direct pipeline use karo - tokenizers avoid karo
            textgen_pipeline = pipeline(
                "text-generation",
                model="distilgpt2",
                device=self.device
            )
            self.models['text_generation'] = textgen_pipeline
            return "✅ Text generation model loaded successfully!"
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def load_image_generation_model(self):
        """Image generation - CPU friendly"""
        try:
            # Lightweight model use karo
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            pipe = pipe.to("cpu")
            self.models['image_generation'] = pipe
            return "✅ Image generation model loaded successfully!"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# Global instance
model_loader = SimpleModelLoader()