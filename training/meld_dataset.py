from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
import cv2
import numpy as np
import torch
import subprocess
import torchaudio
class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
       self.data = pd.read_csv(csv_path)

       self.video_dir = video_dir

       self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

       self.emotion_map = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'joy': 3,
            'neutral': 4,
            'sadness': 5,
            'surprise': 6
       }
       
       self.sentiment_map = {
            'negative': 0,
            'neutral': 1,
            'positive': 2,
       }
    
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            if not cap.isOpened():
                raise ValueError(f"Video capture is not opened: {video_path}")
            
            # try and read first frame to validate video
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video capture is not opened: {video_path}")
            
            # Reset indedx to not skip first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frames.append(frame)
                
                
        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()
        
        if (len(frames) == 0):
            raise ValueError(f"No video captured")
        
        # pad or truncate frames to 30
        if len(frames) < 30:
            frames += [np.zeros_like(frame[0])] * (30 - len(frames))
        else:
            frames = frames[:30]
        
        # before permute : [frames, height, width, channels]
        # after permute : [frames, channels, height, width]
        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)
    
    def _extract_audio_features(self, video_path):
        audio_path = video_path.replace(".mp4", ".wav")
        
        try:
            subprocess.run([ 
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
                
                ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            if sample_rate!= 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512,
            )
            
            mel_spec = mel_spectrogram(waveform)
            
            # Normalize mel spectrogram
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()
            
            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]
            return mel_spec
                
        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
            
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        
        finally: 
            if os.path.exists(audio_path):
                os.remove(audio_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        row = self.data.iloc[idx]
        
        try:    
            row = self.data.iloc[idx]
            video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
            path = os.path.join(self.video_dir, video_filename)
            video_path_exist = os.path.exists(path)
            
            if video_path_exist == False:
                raise FileNotFoundError(f"File {path} not found")
                
            text_inputs = self.tokenizer(row["Utterance"], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
            
            video_frames = self._load_video_frames(path)
            audio_features = self._extract_audio_features(path)
            
            # Map sentiment and emotion labels
            emotion_label = self.emotion_map[row["Emotion"].lower()]
            sentiment_label = self.sentiment_map[row["Sentiment"].lower()]
            
            return {
                'text_inputs':{
                    'input_ids': text_inputs["input_ids"].squeeze(),
                    'attention_mask': text_inputs["attention_mask"].squeeze()
                },
                'video_frames': video_frames,
                'audio_features': audio_features,
                'emotion_label': torch.tensor(emotion_label),
                'sentiment_label': torch.tensor(sentiment_label)
            }
            
            # print(video_frames)
        except Exception as e:
            print(f"Error Processing: {path} {str(e)}")
            return None
        
def prepare_dataloaders(train_csv, train_video_dir, 
                        dev_csv, dev_video_dir,
                        test_csv, test_video_dir, batch_size=32):
    train_dataset = MELDDataset(train_csv, train_video_dir)
    dev_dataset = MELDDataset(dev_csv, dev_video_dir)
    test_dataset = MELDDataset(test_csv, test_video_dir)
    

if __name__ == "__main__":
    meld = MELDDataset("../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete")
    print(meld[0])
    