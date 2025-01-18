from torch.utils.data import Dataset
import pandas as pd
from transformers import AutoTokenizer
import os
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
       
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_filename = f"""dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"""
        path = os.path.join(self.video_dir, video_filename)
        video_path = os.path.exists(path)
        
        if video_path == False:
            raise FileNotFoundError(f"File {path} not found")
            
        print("Video path exists")

if __name__ == "__main__":
    meld = MELDDataset("../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete")
    print(meld[0])
    