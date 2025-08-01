from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from Voice.main import AudioProcessor
from Hand_LandMarks.test import GestureProcessor
from Speech.main import TextRequest
import numpy as np
import torch
import logging
import lameenc
from datetime import datetime
import os
import io
from gtts import gTTS
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],   
)
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    processor = AudioProcessor(ws, loop=loop)

    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(processor.sr)
    encoder.set_channels(1)
    encoder.set_quality(5)
    output_dir = r"C:\Users\TIN\Desktop\Trick"
    os.makedirs(output_dir, exist_ok=True)
    logger.info("WebSocket connection established")
    
    try:
        while True:
            chunk = await ws.receive_bytes()
            chunk_np = np.frombuffer(chunk, dtype = np.int16)
            rms_chunk = np.sqrt(np.mean((chunk_np.astype(np.float32) / 32768.0) ** 2))
            
            mp3_data = encoder.encode(chunk_np.tobytes())
            if mp3_data:
                dt = datetime.now().strftime("%Y%m%d_%H%M%S")  
                file_path = os.path.join(output_dir, f"audio_{dt}.mp3")
                with open(file_path, "ab") as f:  
                    f.write(mp3_data)    
            
            if rms_chunk >= 0.01:
                logger.info("Đã thêm âm thanh vào chunk")
                processor.add_chunk(chunk)
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        mp3_data = encoder.flush()
        if mp3_data:
            dt = datetime.now().strftime("%Y%m%d_%H%M%S")  
            file_path = os.path.join(output_dir, f"audio_final_{dt}.mp3")
            with open(file_path, "ab") as f:
                f.write(mp3_data)
        if processor.audio_buffer.size > 0:
            processor.process_full_buffer()
        processor.stop()
        
    except Exception as e:
        logger.exception("Unexpected error")
        await ws.close(code=1011, reason=str(e))

@app.websocket("/gesture")
async def websocket_endpoint(websocket: WebSocket):
    manager = GestureProcessor()
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.process_data(websocket, data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/generate-speech/")
async def generate_speech(request: TextRequest):
    try:
        tts = gTTS(text=request.text, lang='vi', slow=False)
        audio_file = io.BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)

        # Điều chỉnh tốc độ (nếu cần)
        if request.speed != 1.0:
            pass  # Xử lý tốc độ nếu cần

        headers = {
            "Access-Control-Allow-Origin": "*",
            "Content-Disposition": "attachment; filename=speech.mp3"
        }

        return StreamingResponse(
            audio_file,
            media_type="audio/mpeg",
            headers=headers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")
    