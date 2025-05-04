from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from utils.audio import transcribe_audio
from utils.vision import extract_text_from_image
from utils.prompts import build_prompt
import openai
import os

openai.api_key = "EMPTY"
openai.api_base = "http://local:8000/v1"  # or localhost

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process(file: UploadFile = File(...), slide: UploadFile = File(None)):
    os.makedirs("temp", exist_ok=True)
    audio_path = f"temp/{file.filename}"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    transcript = transcribe_audio(audio_path)

    slide_text = ""
    if slide:
        slide_path = f"temp/{slide.filename}"
        with open(slide_path, "wb") as f:
            f.write(await slide.read())
        slide_text = extract_text_from_image(slide_path)

    prompt = build_prompt(transcript, slide_text)

    response = openai.ChatCompletion.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are an AI teaching assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return {"annotations": response["choices"][0]["message"]["content"]}
