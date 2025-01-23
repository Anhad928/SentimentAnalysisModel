from collections import namedtuple
import torch
from torch.utils.data import DataLoader
from training.models import MultimodalSentimentModel, MultimodalTrainer

def test_logging():
    Batch = namedtuple('Batch', ['text_inputs', 'video_frames', 'audio_features'])
    
    mock_batch = Batch(text_inputs={'input_ids': torch.ones(1), 'attention_mask': torch.ones(1)},
                       video_frames=torch.ones(1),
                       audio_features=torch.ones(1)
                       )
    mock_loader = DataLoader([mock_batch])
    
    model = MultimodalSentimentModel()
    trainer = MultimodalTrainer(model, mock_loader, mock_loader)
    
if __name__ == '__main__':
    test_logging()
    