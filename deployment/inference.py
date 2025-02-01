import torch
from models import MultimodalSentimentModel
import os

def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalSentimentModel().to(device)
    
    model_path = os.path.join(model_dir, "model.pth")