import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from collections import deque
import numpy as np
import json
import time
from typing import List, Dict

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("Hand_LandMarks/hand_gesture_model_relative.pth", map_location=device)
GESTURE_CLASSES = checkpoint['classes']
SEQUENCE_LENGTH = checkpoint['sequence_length']
NUM_FEATURES_PER_HAND = 21 * 3
INPUT_SIZE = 2 * NUM_FEATURES_PER_HAND

class LabelEncoder:
    def __init__(self, classes):
        self.classes = classes
        self.class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        self.index_to_class = {idx: cls for idx, cls in enumerate(classes)}
    
    def transform(self, labels):
        return [self.class_to_index[label] for label in labels]
    
    def inverse_transform(self, indices):
        return [self.index_to_class[idx] for idx in indices]

le = LabelEncoder(GESTURE_CLASSES)

class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=INPUT_SIZE, hidden_size=128, num_classes=len(GESTURE_CLASSES)):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.dropout1 = torch.nn.Dropout(0.3)
        self.lstm2 = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(hidden_size // 2, 64)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, num_classes)
    
    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = LSTMModel().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

class GestureProcessor:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.sequence_buffers = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.sequence_buffers[websocket] = deque(maxlen=SEQUENCE_LENGTH)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        del self.sequence_buffers[websocket]

    async def process_data(self, websocket: WebSocket, data: dict):
        # Thêm dữ liệu vào bộ đệm
        features = data["features"]
        sequence_buffer = self.sequence_buffers[websocket]
        sequence_buffer.append(features)
        
        # Xử lý khi đủ chuỗi
        if len(sequence_buffer) >= SEQUENCE_LENGTH:
            sequence = list(sequence_buffer)[-SEQUENCE_LENGTH:]
            input_tensor = torch.tensor([sequence], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()
                gesture = le.inverse_transform([predicted_idx])[0]
            
            # Gửi kết quả về client
            response = {
                "gesture": gesture,
                "confidence": confidence,
                "timestamp": time.time()
            }
            await websocket.send_json(response)
