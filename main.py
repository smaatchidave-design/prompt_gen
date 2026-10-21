import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles  #  NEW: Handles separate asset distribution
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

app = FastAPI()

#  NEW: Connects the /static web path directly to your static asset folder
app.mount("/static", StaticFiles(directory="static"), name="static")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class PromptRequest(BaseModel):
    user_input: str
    role: str
    tone: str

@app.get("/")
async def serve_homepage():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"Error": "index.html file was not found in this folder!"}

@app.post("/generate-prompt")
async def generate_prompt(request: PromptRequest):
    if not request.user_input.strip():
        raise HTTPException(status_code=400, detail="Input cannot be empty")
        
    system_instruction = (
        "You are an elite Prompt Engineer. Your task is to transform the user's raw idea "
        "into a structured, high-performing system prompt. "
        f"Design the prompt as if it is written for a {request.role} acting in a {request.tone} manner. "
        "Structure the output with sections like [Context], [Role], [Task Instructions], and [Expected Output]. "
        "Output ONLY the finalized prompt text. Do not include chat intro or outro text."
        "Then at the end, say '🍌 DEVELOPED BY DAVID'"
    )

    try:
        completion = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Raw Idea: {request.user_input}"}
            ],
            temperature=0.6,
        )
        return {"smart_prompt": completion.choices[0].message.content}

    except Exception as e:
        print(f"GROQ API ERROR DETECTED: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))