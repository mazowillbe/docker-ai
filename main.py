from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Qwen-3 1.7B Instruct API")

# Load Qwen-3 1.7B Instruct at startup
generator = pipeline("text-generation", model="Qwen/Qwen-3-1.7B-Instruct")

class Prompt(BaseModel):
    text: str
    max_length: int = 128

@app.get("/")
def root():
    return {"message": "Qwen-3 1.7B API running. Visit /docs for Swagger UI."}

@app.post("/generate")
def generate(prompt: Prompt):
    output = generator(prompt.text, max_length=prompt.max_length, num_return_sequences=1)
    return {"input": prompt.text, "output": output[0]["generated_text"]}
