import os
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F


warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r".*torch\.utils\._pytree\._register_pytree_node is deprecated.*"
)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer once
model_path = "./model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# FastAPI app
app = FastAPI(title="Text Classification API")

# Request schema
class TextInput(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict(input: TextInput):
    try:
        inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label].item()
        return {"label": label, "confidence": round(confidence, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
