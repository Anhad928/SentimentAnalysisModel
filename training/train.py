import os
import argparse
import torchaudio
import torch
import tqdm
import json

from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer

# AWS SageMaker
SM_MODEL_DIR = os.environ.get('SM_MODEL_DIR', '.')
SM_CHANNEL_TRAINING = os.environ('SM_CHANNEL_TRAIN', '/opt/ml/input/data/training')
SM_CHANNEL_VALIDATION = os.environ('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation')
SM_CHANNEL_TEST = os.environ('SM_CHANNEL_TEST', '/opt/ml/input/data/test')

os.environ['PYTORCH_CUDE_ALLOC_CONF'] = 'expandable_segment:True'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    
    # Data directories
    parser.add_argument("--train-dir", type=str, default=SM_CHANNEL_TRAINING)
    parser.add_argument("--val-dir", type=str, default=SM_CHANNEL_VALIDATION)
    parser.add_argument("--test-dir", type=str, default=SM_CHANNEL_TEST)
    parser.add_argument("--model-dir", type=str, default=SM_MODEL_DIR)
    
    return parser.parse_args()

def main():
    # Install FFMPEG
    print("Available audio backends:")
    print(str(torchaudio.list_audio_backends()))
    
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Track inintial GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated()/ 104**3
        print("Initial GPU memory:")
        print(f"Initial memory used {memory_used:.2f} GB")
        
    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'output_repeated_splits.csv'),
        batch_size=args.batch_size
    )
    
    print(f"""Trainig DSV path: {os.path.join(
        args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training video Directory: {
        os.path.join(args.train_dir, 'train_splits')}""")
    
    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    best_val_loss = float('inf')
    
    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": []
        
    }
    
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        trainer_loss = trainer.train_epoch()
        val_loss, val_matrics = trainer.evaluate(val_loader)
        
        # Track metrics
        metrics_data['train_losses'].append(trainer_loss['total'])
        metrics_data['val_losses'].append(val_loss['total'])
        metrics_data['epochs'].append(epoch)
        
        #Log metrics in SageMaker Format
        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": trainer_loss['total']},
                {"Name": "validation:loss", "Value": val_loss['total']},
                {"Name": "validation:emotion_precision", "Value": val_matrics['emotion_precision']},
                {"Name": "validation:emotion_accuracy", "Value": val_matrics['emotion_accuracy']},
                {"Name": "validation:sentiment_precision", "Value": val_matrics['sentiment_precision']},
                {"Name": "validation:sentiment_accuracy", "Value": val_matrics['sentiment_accuracy']}
            ]
        }))
        
        if torch.cuda.is_available():
            # torch.cuda.reset_peak_memory_stats()
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Initial GPU memory used: {memory_used:.2f} GB")
            
        # Save best model
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'model.pth'))
            
            #After training is complete, evaluate on test set
            print("Evaluating on test set...")
            test_loss, test_metrics = trainer.evaluate(test_loader, phase='test')
            metrics_data['test_loss'] = test_loss['total']
            
            print(json.dumps({
                "metrics": [
                    {"Name": "test:loss", "Value": test_loss["total"]},
                    {"Name": "test:emotion_accuracy",
                        "Value": test_metrics["emotion_accuracy"]},
                    {"Name": "test:sentiment_accuracy",
                        "Value": test_metrics["sentiment_accuracy"]},
                    {"Name": "test:emotion_precision",
                        "Value": test_metrics["emotion_precision"]},
                    {"Name": "test:sentiment_precision",
                        "Value": test_metrics["sentiment_precision"]},
                ]
            }))

            
        
        
    
if __name__ == '__main__':
    main()
