from faster_whisper import WhisperModel, download_model
from pyannote.audio import Pipeline
import dotenv
import os
import torch
from huggingface_hub import login
import dotenv

token = dotenv.get_key("Token_Hugginface")
login(token=token)
dotenv.load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# name_model = "Voice/base/whisper-small"
name_model = "Voice/models/whisper-medium"
# download_model("medium", output_dir="models/whisper-medium")
whisper = WhisperModel(name_model, device="cuda", compute_type="float16", cpu_threads=4, num_workers=2)
