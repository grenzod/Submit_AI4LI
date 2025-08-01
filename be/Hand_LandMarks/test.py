import time
from collections import deque
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

SEQUENCE_LENGTH   = 35
NUM_JOINTS        = 21
COORDS_PER_JOINT  = 3
INPUT_SIZE        = 2 * NUM_JOINTS * COORDS_PER_JOINT  # = 126
HIDDEN_SIZES      = (256, 128, 64)
DROPOUT           = 0.3

class GestureModel(nn.Module):
    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_sizes: tuple = HIDDEN_SIZES,
        num_classes: int = 6,
        dropout: float = DROPOUT
    ):
        super().__init__()
        h1, h2, h3 = hidden_sizes

        self.lstm1  = nn.LSTM(input_size, h1, batch_first=True)
        self.ln1    = nn.LayerNorm(h1)
        self.do1    = nn.Dropout(dropout)

        self.lstm2  = nn.LSTM(h1, h2, batch_first=True)
        self.ln2    = nn.LayerNorm(h2)
        self.do2    = nn.Dropout(dropout)

        self.lstm3  = nn.LSTM(h2, h3, batch_first=True)
        self.ln3    = nn.LayerNorm(h3)
        self.do3    = nn.Dropout(dropout)

        self.fc1     = nn.Linear(h3, 128)
        self.bn_fc1  = nn.BatchNorm1d(128)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2     = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        x, _ = self.lstm1(x)    # → (B, T, h1)
        x     = self.ln1(x)
        x     = self.do1(x)

        x, _ = self.lstm2(x)    # → (B, T, h2)
        x     = self.ln2(x)
        x     = self.do2(x)

        x, _ = self.lstm3(x)    # → (B, T, h3)
        x     = self.ln3(x)
        x     = self.do3(x)

        x = x[:, -1, :]         # → (B, h3)
        x = self.fc1(x)         # → (B, 128)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)         # → (B, num_classes)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("Hand_LandMarks/best_hand_gesture_model.pth", map_location=device)

mu      = ckpt["mu"]            # shape (INPUT_SIZE,)
sigma   = ckpt["sigma"]         # shape (INPUT_SIZE,)
sk_le   = ckpt["label_encoder"] # a sklearn LabelEncoder
classes = list(sk_le.classes_)

class LabelEncoder:
    def __init__(self, classes: List[str]):
        self.index_to_class = {i: cl for i, cl in enumerate(classes)}

    def inverse_transform(self, idxs: List[int]) -> List[str]:
        return [self.index_to_class[i] for i in idxs]

le = LabelEncoder(classes)


model = GestureModel(
    input_size=INPUT_SIZE,
    hidden_sizes=HIDDEN_SIZES,
    num_classes=len(classes),
    dropout=DROPOUT
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def normalize_hand(hand_data):
    if hand_data is None or hand_data[0]["x"] < 0:
        return [0.0] * (NUM_JOINTS * COORDS_PER_JOINT)

    wrist   = hand_data[0]
    mid_tip = hand_data[12] 

    dx = mid_tip["x"] - wrist["x"]
    dy = mid_tip["y"] - wrist["y"]
    dz = mid_tip["z"] - wrist["z"]
    scale = max(np.linalg.norm([dx, dy, dz]), 1e-3)

    normalized = []
    for p in hand_data:
        normalized.append((p["x"] - wrist["x"]) / scale)
        normalized.append((p["y"] - wrist["y"]) / scale)
        normalized.append((p["z"] - wrist["z"]) / scale)

    return normalized

class GestureProcessor:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.buffers: Dict[WebSocket, deque] = {}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.append(ws)
        self.buffers[ws] = deque(maxlen=SEQUENCE_LENGTH)

    def disconnect(self, ws: WebSocket):
        self.active_connections.remove(ws)
        del self.buffers[ws]

    async def process_data(self, ws: WebSocket, data: dict):
        left_vec  = normalize_hand(data["left_hand"])
        right_vec = normalize_hand(data["right_hand"])

        raw  = np.array(left_vec + right_vec, dtype=np.float32)
        norm = (raw - mu) / sigma

        buf = self.buffers[ws]
        buf.append(norm)

        if len(buf) == SEQUENCE_LENGTH:
            seq = np.stack(buf, axis=0)           # (T, 126)
            x   = torch.from_numpy(seq[None]).to(device)  # (1, T, 126)

            with torch.no_grad():
                logits = model(x)                # (1, num_classes)
                probs  = torch.softmax(logits, dim=1)
                conf, idx = probs.max(dim=1)

                gesture    = le.inverse_transform([idx.item()])[0]
                confidence = conf.item()

            await ws.send_json({
                "gesture":    gesture,
                "confidence": confidence,
                "timestamp":  time.time()
            })
