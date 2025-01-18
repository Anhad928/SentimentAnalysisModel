from torch.utils.data import Dataset
import pandas as pd
class MELDDataset(Dataset):
    def __init__(self, csv_path, video_dir):
       self.data = pd.read_csv(csv_path)
       print(len(self.data))

if __name__ == "__main__":
    meld = MELDDataset("../dataset/dev/dev_sent_emo.csv", "../dataset/dev/dev_splits_complete")