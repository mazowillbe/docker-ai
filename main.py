from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, AutoTokenizer
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen-3 1.7B API",
    description="FastAPI wrapper for Qwen-3-1.7B model",
    version="1.0.0"
)

# Get token from environment
HF_TOKEN = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not HF_TOKEN:
    logger.warning("HUGGINGFACE_HUB_TOKEN not found in environment variables")

# Model configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"

try:
    # Initialize tokenizer and pipeline
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=HF_TOKEN,
        trust_remote_code=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize pipeline with optimizations
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        use_auth_token=HF_TOKEN,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        model_kwargs={
            "load_in_8bit": True,  # Enable 8-bit quantization for memory efficiency
            "low_cpu_mem_usage": True
        }
    )
    
    logger.info("Model loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    generator = None

class Prompt(BaseModel):
    text: str = Field(..., description="Input text for generation")
    max_length: int = Field(default=512, ge=1, le=32768, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.1, le=1.0, description="Top-p sampling parameter")
    do_sample: bool = Field(default=True, description="Enable sampling")

@app.get("/")
def root():
    return {
        "message": "Qwen-3 1.7B API running",
        "model": MODEL_NAME,
        "status": "ready" if generator else "error"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if generator else "unhealthy",
        "model_loaded": generator is not None
    }

@app.post("/generate")
def generate(prompt: Prompt):
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details."
        )
    
    try:
        # Prepare messages for chat format
        messages = [{"role": "user", "content": prompt.text}]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        outputs = generator(
            text,
            max_new_tokens=prompt.max_length,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            do_sample=prompt.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Extract only the generated response
        response_text = outputs[0]["generated_text"]
        
        return {
            "input": prompt.text,
            "output": response_text,
            "parameters": {
                "max_length": prompt.max_length,
                "temperature": prompt.temperature,
                "top_p": prompt.top_p
            }
        }
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_completion(messages: list[dict[str, str]], prompt: Prompt):
    """
    Full chat completion endpoint that accepts conversation history
    """
    if not generator:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs for details."
        )
    
    try:
        # Convert messages to the required format
        conversation = [{"role": msg["role"], "content": msg["content"]} 
                       for msg in messages]
        
        # Apply chat template
        text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        outputs = generator(
            text,
            max_new_tokens=prompt.max_length,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            do_sample=prompt.do_sample,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        response_text = outputs[0]["generated_text"]
        
        return {
            "messages": conversation,
            "response": response_text,
            "parameters": {
                "max_length": prompt.max_length,
                "temperature": prompt.temperature,
                "top_p": prompt.top_p
            }
        }
        
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
