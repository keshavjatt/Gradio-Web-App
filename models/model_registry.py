import torch
import torch._dynamo
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import yaml
import os
from typing import Dict, Any, Optional
import logging
import gc
from functools import lru_cache

# Optimize torch for CPU
torch.set_num_threads(4)
torch._dynamo.config.suppress_errors = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedModelRegistry:
    def __init__(self, config_path: str = "configs/app_config.yaml"):
        self.config = self._load_config(config_path)
        self.device = self.config['models']['device']
        self.models = {}
        self.tokenizers = {}
        self._setup_optimizations()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_optimizations(self):
        """Setup PyTorch optimizations for CPU"""
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('medium')
    
    def _optimize_diffusion_pipeline(self, pipe):
        """Apply optimizations to diffusion pipeline - CPU COMPATIBLE"""
        try:
            # Use faster scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, 
                use_karras_sigmas=True,
                algorithm_type="sde-dpmsolver++"
            )
            
            # Enable attention slicing for memory efficiency
            if self.config['optimization']['use_attention_slicing']:
                pipe.enable_attention_slicing()
            
            # Enable CPU offloading
            if self.config['optimization']['model_offload']:
                pipe.enable_sequential_cpu_offload()
            
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")
        
        return pipe
    
    def load_sentiment_analysis(self):
        """Load optimized sentiment analysis model"""
        if 'sentiment' in self.models:
            return self.models['sentiment']
            
        try:
            model_name = self.config['models']['sentiment_analysis']['model']
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=-1,
                torch_dtype=torch.float32
            )
            self.models['sentiment'] = sentiment_pipeline
            logger.info(f"✅ Loaded sentiment model: {model_name}")
            return sentiment_pipeline
        except Exception as e:
            logger.error(f"❌ Error loading sentiment model: {e}")
            return None
    
    def load_summarization(self):
        """Load optimized summarization model"""
        if 'summarization' in self.models:
            return self.models['summarization']
            
        try:
            model_name = self.config['models']['summarization']['model']
            summarization_pipeline = pipeline(
                "summarization",
                model=model_name,
                device=-1,
                torch_dtype=torch.float32
            )
            self.models['summarization'] = summarization_pipeline
            logger.info(f"✅ Loaded summarization model: {model_name}")
            return summarization_pipeline
        except Exception as e:
            logger.error(f"❌ Error loading summarization model: {e}")
            return None
    
    def load_next_word_prediction(self):
        """Load optimized next word prediction model"""
        if 'next_word' in self.models:
            return self.models['next_word'], self.tokenizers.get('next_word')
            
        try:
            model_name = self.config['models']['next_word_prediction']['model']
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if missing
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model.eval()
            
            self.models['next_word'] = model
            self.tokenizers['next_word'] = tokenizer
            logger.info(f"✅ Loaded next-word model: {model_name}")
            return model, tokenizer
        except Exception as e:
            logger.error(f"❌ Error loading next-word model: {e}")
            return None, None
    
    def load_translation(self):
        """Load optimized translation model"""
        if 'translation' in self.models:
            return self.models['translation']
            
        try:
            model_name = self.config['models']['translation']['model']
            translation_pipeline = pipeline(
                "translation_en_to_fr",
                model=model_name,
                device=-1,
                torch_dtype=torch.float32
            )
            self.models['translation'] = translation_pipeline
            logger.info(f"✅ Loaded translation model: {model_name}")
            return translation_pipeline
        except Exception as e:
            logger.error(f"❌ Error loading translation model: {e}")
            return None
    
    def load_ner(self):
        """Load optimized named entity recognition model"""
        if 'ner' in self.models:
            return self.models['ner']
            
        try:
            model_name = self.config['models']['ner']['model']
            ner_pipeline = pipeline(
                "ner",
                model=model_name,
                device=-1,
                torch_dtype=torch.float32,
                aggregation_strategy="simple"
            )
            self.models['ner'] = ner_pipeline
            logger.info(f"✅ Loaded NER model: {model_name}")
            return ner_pipeline
        except Exception as e:
            logger.error(f"❌ Error loading NER model: {e}")
            return None
    
    def load_image_generation(self):
        """Load CPU-optimized image generation model"""
        if 'image_gen' in self.models:
            return self.models['image_gen']
            
        try:
            model_name = self.config['models']['image_generation']['model']
            
            logger.info("🚀 Loading CPU-optimized image generation model...")
            
            pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Apply CPU optimizations
            pipe = self._optimize_diffusion_pipeline(pipe)
            
            self.models['image_gen'] = pipe
            logger.info(f"✅ Loaded image generation model: {model_name}")
            return pipe
            
        except Exception as e:
            logger.error(f"❌ Error loading image generation model: {e}")
            return None
    
    def unload_model(self, model_key: str):
        """Unload a specific model to free memory"""
        if model_key in self.models:
            del self.models[model_key]
            if model_key in self.tokenizers:
                del self.tokenizers[model_key]
            
            gc.collect()
            logger.info(f"🗑️ Unloaded model: {model_key}")
    
    def cleanup(self):
        """Cleanup all models and free memory"""
        self.models.clear()
        self.tokenizers.clear()
        gc.collect()
        logger.info("🧹 Cleaned up all models")