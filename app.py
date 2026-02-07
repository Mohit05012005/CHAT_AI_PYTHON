from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
from typing import Union

# Load environment variables
load_dotenv()

# Initialize model
api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    groq_api_key=api_key,
    model="llama-3.3-70b-versatile"
)

# FastAPI app
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://chat-app-frontend-two-rho.vercel.app/",
        "https://chat-app-backend-ahq0.onrender.com/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class Question(BaseModel):
    qns: str

# Root
@app.get("/")
async def root():
    res = await model.ainvoke("Hello")
    return {"message": res.content}

# Simple chat (GET)
@app.get("/ai/chat/{item_id}")
async def chat(item_id: int, q: Union[str, None] = None):
    if not q:
        return {"error": "Query missing"}

    res = await model.ainvoke(q)
    return {
        "item_id": item_id,
        "answer": res.content
    }

# Main API (POST)
@app.post("/api/qns-ans")
async def ask_ai(item: Question):
    try:
        res = await model.ainvoke(item.qns)
        return {"answer": res.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
