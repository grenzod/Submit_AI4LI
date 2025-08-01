from Voice.base.models import whisper  
import threading
import queue
import numpy as np
import asyncio
import logging
import json
import noisereduce as nr
import scipy.signal
import wave
import os
from datetime import datetime
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIETNAMESE_CONFIG = {
    "language": "vi",
    "beam_size": 5,
    "vad_options": {
        "threshold": 0.45,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 300,
        "speech_pad_ms": 30
    },
    "transcription_options": {
        "no_speech_threshold": 0.5,
        "log_prob_threshold": -0.8,
        "compression_ratio_threshold": 2.2,
        "condition_on_previous_text": False,
        "repetition_penalty": 1.2,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": "Xin chào, đây là hệ thống nhận dạng giọng nói tiếng Việt."
    },
    "audio_processing": {
        "target_db": -20.0,
        "noise_reduction_strength": 0.75,
        "high_pass_freq": 100,
        "low_pass_freq": 4000
    }
}

def enhance_audio(samples_np, sr=16000):
    try:
        # Chuẩn hóa âm lượng
        rms = np.sqrt(np.mean(samples_np**2))
        if rms > 0:
            target_rms = 10 ** (VIETNAMESE_CONFIG["audio_processing"]["target_db"] / 20)
            samples_np = samples_np * (target_rms / rms)
        
        # Khử nhiễu
        noise_sample = samples_np[:int(sr * 0.1)] 
        samples_np = nr.reduce_noise(
            y=samples_np, 
            sr=sr, 
            stationary=False,
            y_noise=noise_sample,
            prop_decrease=VIETNAMESE_CONFIG["audio_processing"]["noise_reduction_strength"]
        )
        
        # Bộ lọc thông dải
        nyquist = 0.5 * sr
        low = VIETNAMESE_CONFIG["audio_processing"]["high_pass_freq"] / nyquist
        high = VIETNAMESE_CONFIG["audio_processing"]["low_pass_freq"] / nyquist
        b, a = scipy.signal.butter(4, [low, high], btype='band')
        samples_np = scipy.signal.filtfilt(b, a, samples_np)
        
        return samples_np.astype(np.float32)
    except Exception as e:
        logger.error(f"Audio enhancement failed: {str(e)}")
        return samples_np.astype(np.float32)

def bytes_to_audiosegment(audio_bytes, sr=16000):
    wav = AudioSegment(
        data=audio_bytes,
        sample_width=2,
        frame_rate=sr,
        channels=1
    )
    return wav

class AudioProcessor:
    def __init__(self, ws, loop, sr=16000, min_duration=10.0, retain_seconds=0.8):
        self.ws = ws
        self.loop = loop
        self.sr = sr
        self.min_samples = int(sr * min_duration)
        self.retain_samples = int(sr * retain_seconds)
        self.retain_bytes = int(retain_seconds * sr * 2)
        self.audio_buffer = np.empty((0,), dtype=np.float32)
        self.chunk_queue = queue.Queue()
        self.active = True

        self.lock = threading.Lock()
        self.can_spawn = True
        self.processing_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.processing_thread.start()
        logger.info("Audio processor initialized")

    def add_chunk(self, chunk):
        if not self.active:
            return
        
        val = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            self.chunk_queue.put_nowait(val)
        except queue.Full:
            logger.warning("Audio queue full, dropping chunk")

    def process_audio(self):
        while self.active:
            try:
                samples = self.chunk_queue.get(timeout=0.1)
                self.audio_buffer = np.concatenate([self.audio_buffer, samples])
                self.chunk_queue.task_done()
                
                if len(self.audio_buffer) >= self.min_samples and self.can_spawn:
                    with self.lock:
                        self.can_spawn = False
                        threading.Thread(target=self.process_full_buffer, daemon=True).start()
            except queue.Empty:
                continue

    def process_full_buffer(self):
        to_process = self.audio_buffer[:self.min_samples].copy()
        
        if len(self.audio_buffer) > self.retain_samples:
            self.audio_buffer = self.audio_buffer[-self.retain_samples:].copy()
        else:
            self.audio_buffer = np.empty((0,), dtype=np.float32)
        
        self.can_spawn = True        
        self._save_wav_debug(to_process)       
        self.transcribe_audio(to_process)

    def _save_wav_debug(self, samples: np.ndarray):
        debug_dir = r"C:\Users\TIN\Desktop\Trick\Voice_Test"
        os.makedirs(debug_dir, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(debug_dir, f"audio_{ts}.wav")
        
        # Chuẩn int16
        int16_data = (samples * 32767).astype(np.int16)
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sr)
            wf.writeframes(int16_data.tobytes())
        
        duration = len(samples) / self.sr
        logger.info(f"Saved {duration:.1f}s debug audio: {path}")

    def transcribe_audio(self, samples_np: np.ndarray):
        """Nhận dạng âm thanh (chạy trong thread riêng)"""
        try:
            logger.info("Bắt đầu nhận dạng...")
            samples_np = enhance_audio(samples_np, self.sr)
            
            options = {
                "language": VIETNAMESE_CONFIG["language"],
                "beam_size": VIETNAMESE_CONFIG["beam_size"],
                "vad_filter": True,
                "vad_parameters": VIETNAMESE_CONFIG["vad_options"],
                "task": "transcribe",
                "word_timestamps": True
            }
            options.update(VIETNAMESE_CONFIG["transcription_options"])
            
            segments, _ = whisper.transcribe(
                samples_np,
                **options
            )
            
            transcript = " ".join(seg.text.strip() for seg in segments).strip()
            audio_duration = len(samples_np) / self.sr
            
            if transcript:
                logger.info(f"🎤 Nhận dạng thành công ({audio_duration:.1f}s): {transcript}")
                msg = json.dumps({"transcript": transcript, "is_final": True})
                self.loop.call_soon_threadsafe(
                    asyncio.create_task,
                    self.ws.send_text(msg)
                )

            else:
                logger.warning(f"🔇 Không phát hiện giọng nói trong {audio_duration:.1f}s")
            
            # Giữ lại phần âm thanh cuối
            with self.lock:
                if len(self.audio_buffer) > self.retain_samples:
                    self.audio_buffer = self.audio_buffer[-self.retain_samples:].copy()
                else:
                    self.audio_buffer = np.empty((0,), dtype=np.float32)
                    
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            with self.lock:
                self.audio_buffer = np.empty((0,), dtype=np.float32)

    def stop(self):
        self.active = False
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        logger.info("Audio processor stopped")
        