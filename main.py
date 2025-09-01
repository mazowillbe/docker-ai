from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Tiny GPT-2 API")

# Load model once at startup
generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

class Prompt(BaseModel):
    text: str
    max_length: int = 50

@app.get("/")
def root():
    return {"message": "Tiny GPT-2 API is running. Visit /docs for Swagger UI."}

@app.post("/generate")
def generate(prompt: Prompt):
    output = generator(prompt.text, max_length=prompt.max_length, num_return_sequences=1)
    return {"input": prompt.text, "output": output[0]["generated_text"]}
