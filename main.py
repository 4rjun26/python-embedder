from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
from translatepy import Translator  # Import translatepy

# Load the model only once
model = SentenceTransformer('all-MiniLM-L6-v2')
translator = Translator()  # Initialize the translator

app = FastAPI()

# Request body structure for embedding
class EmbedRequest(BaseModel):
    text: str

# Response structure for embedding
class EmbedResponse(BaseModel):
    embedding: list

@app.post("/embed", response_model=EmbedResponse)
async def get_embedding(req: EmbedRequest):
    try:
        embedding = model.encode(req.text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Request body structure for translation
class TranslateRequest(BaseModel):
    text: str
    target_language: str

# Response structure for translation
class TranslateResponse(BaseModel):
    translated_text: str

@app.post("/translate", response_model=TranslateResponse)
async def translate_text(req: TranslateRequest):
    try:
        # Translate the text using translatepy
        translated = translator.translate(req.text, req.target_language)
        return {"translated_text": translated.result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# For local testing
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
