import os
from tqdm import tqdm
import torch
from dataclasses import dataclass

@dataclass
class TrainerParams:
    epochs: int
    patience: int
    device: torch.device
    save_dir: str
    save_prefix: str

class Trainer:
    def __init__(self, model, loss, optimizer, train_dataloader, val_dataloader, params: TrainerParams):
        #data
        self.tr_dl = train_dataloader
        self.val_dl = val_dataloader
        
        #params
        self.trainer_params = params
        self.save_dir = params.save_dir
        self.save_prefix = params.save_prefix
        self.patience = params.patience
        self.device = params.device
       
        self.model = model      
        self.loss_fn = loss
        self.optimizer = optimizer

        os.makedirs(self.trainer_params.save_dir, exist_ok=True)

        self.best_loss = float("inf")
        self.not_improved = 0
        self.tr_losses, self.val_losses = [], []

    @staticmethod
    def to_device(batch, device):
        ims, gts = batch
        return ims.to(device), gts.to(device)

    def train_epoch(self):
        self.model.train()
        train_loss  = 0.0

        for idx, batch in tqdm(enumerate(self.tr_dl), desc="Training"):
            
            ims, gts = Trainer.to_device(batch = batch, device = self.trainer_params.device)
            
            # Forward pass
            preds = self.model(ims)
            loss = self.loss_fn(preds, gts)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            train_loss += loss.item()

        train_loss /= len(self.tr_dl)
        
        self.tr_losses.append(train_loss)

        return train_loss

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(self.val_dl), desc="Validation"):
                ims, gts = Trainer.to_device(batch, device = self.trainer_params.device)
                preds = self.model(ims)
                loss = self.loss_fn(preds, gts)

                # Update metrics
                val_loss += loss.item()

        val_loss /= len(self.val_dl)

        self.val_losses.append(val_loss)

        return val_loss

    def save_best_model(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss            
            save_path = os.path.join(self.save_dir, f"{self.save_prefix}_best_model.pth")
            torch.save(self.model.state_dict(), save_path)
            print(f"Best model saved with loss: {self.best_loss:.3f}")
            self.not_improved = 0
        else:
            self.not_improved += 1
            print(f"No improvement for {self.not_improved} epoch(s).")

    def verbose(self, epoch, loss, process = "train"):
        print(f"{epoch + 1}-epoch {process} loss -> {loss:.3f}")
    
    def run(self):
        print("Start training...")

        for epoch in range(self.trainer_params.epochs):
                    
            print(f"\nEpoch {epoch + 1}/{self.trainer_params.epochs}:\n")

            train_loss = self.train_epoch()
            self.verbose(epoch, train_loss, process = "train")

            val_loss = self.validate_epoch()
            self.verbose(epoch, val_loss, process = "validation")            

            self.save_best_model(val_loss)

            if self.not_improved >= self.patience:
                print("Early stopping triggered.")
                break