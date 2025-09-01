from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline, AutoTokenizer
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen-3-1.7B-CPU API",
    description="Lightweight API for Qwen-3-1.7B-INT4 on CPU",
    version="1.0.0"
)

HF_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
MODEL_NAME = "Qwen/Qwen-3-1.7B"  # ← dash, not Qwen3

try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_auth_token=HF_TOKEN,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        use_auth_token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype="auto",
        model_kwargs={
            "low_cpu_mem_usage": True
        }
    )
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Could not load model: {e}")
    generator = None

# ---------- Schemas ----------
class Prompt(BaseModel):
    text: str
    max_length: int = Field(128, ge=1, le=1024)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    top_p: float = Field(0.9, ge=0.1, le=1.0)

class CodePrompt(BaseModel):
    instruction: str
    language: str = "python"
    max_tokens: int = Field(128, ge=1, le=512)

# ---------- Routes ----------
@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME, "loaded": generator is not None}

@app.get("/health")
def health():
    return {"status": "healthy" if generator else "unhealthy"}

@app.post("/generate")
def generate(payload: Prompt):
    if not generator:
        raise HTTPException(503, "Model not loaded")
    messages = [{"role": "user", "content": payload.text}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        outputs = generator(
            prompt_text,
            max_new_tokens=payload.max_length,
            temperature=payload.temperature,
            top_p=payload.top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        return {"input": payload.text, "output": outputs[0]["generated_text"]}
    except Exception as e:
        logger.exception("Generation error")
        raise HTTPException(500, str(e))

@app.post("/generate_code")
def generate_code(payload: CodePrompt):
    if not generator:
        raise HTTPException(503, "Model not loaded")
    system = f"You are a {payload.language} coding assistant."
    prompt = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{payload.instruction}<|im_end|>\n<|im_start|>assistant\n"
    )
    try:
        out = generator(
            prompt,
            max_new_tokens=payload.max_tokens,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        return {"language": payload.language, "generated_code": out[0]["generated_text"]}
    except Exception as e:
        logger.exception("Code generation error")
        raise HTTPException(500, str(e))
