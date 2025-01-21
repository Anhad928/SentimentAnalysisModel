import torch.nn as nn
import torch
from transformers import BertModel
from torchvision import models as vision_models

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.projection = nn.Linear(768, 128)
        
        
        def forward(self, input_ids, attention_mask):
            # Extract Bert embeddings
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            
            #Use [CLS] token representation 
            pooler_output = outputs.pooler_output
            
            return self.projection(pooler_output)
        
        
class VideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        num_fts = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        # [batch_size, framses, height, width] -> [batch_size, channels, frames, height, width]
        x  = x.transport(1,2)
        return self.backbone(x)
    

class AudioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # Lower level feature
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            ## Higher level features
            nn.Conv1d(64,128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        
        for param in self.conv_layer.parameters():
            param.requires_grad = False
        
        self.projection = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def forward(self, x):
        x = x.squeeze(1)
        
        features = self.conv_layer(x)
        # Features output: [batch_size, 128, 1]
        
        return self.projection(features.squeeze(-1))


class MultimodalSentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoders
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(128*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Classification heads
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256, 65),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7) # Sadness, anger
            
        )
        
        self.sentiment_classifier =  nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3) # negative, neutral, positive
        )
        
    def forward(self, text_input, video_frames, audio_fetures):
        text_features = self.text_encoder(
            text_input['input_ids'],
            text_input['attention_mask']
        )
        
        video_features = self.video_encoder(video_frames)
        audio_fetures = self.audio_encoder(audio_fetures)
        
        # concatenate the features
        combined_features = torch.cat([
            text_features,
            video_features,
            audio_fetures
        ], dim=1) # [batch_size, 128 * 3]
        
        
        # FUsion Layer
        
        fused_features = self.fusion_layer(combined_features)
        
        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features)