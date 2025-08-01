from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from gtts import gTTS
import io
from pydantic import BaseModel

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    speed: float = 1.0  # Tốc độ mặc định
