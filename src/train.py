import os
import json
import seaborn as sns
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import numpy as np

class PhoneClassifierTrainer:
    def __init__(
        self,
        data_dir="data/",
        model_path="weights/model.pth",
        img_size=224,
        batch_size=32,
        epochs=50,
        patience=5,
        min_delta=0.0001,
        lr=1e-4,
        class_names=['back', 'front', 'none'],
        num_workers=2,   
        random_seed=42,
    ):
        self.DATA_DIR = data_dir
        self.MODEL_PATH = model_path
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.EARLY_STOPPING_PATIENCE = patience
        self.MIN_DELTA = min_delta
        self.CLASS_NAMES = class_names
        self.NUM_CLASSES = len(class_names)
        self.LR = lr
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.NUM_WORKERS = num_workers
        self.SEED = random_seed

        # For reproducibility
        torch.manual_seed(self.SEED)
        np.random.seed(self.SEED)

        self._init_transforms()
        self._prepare_datasets()
        self._init_model()

    def _init_transforms(self):
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_test_transforms = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _prepare_datasets(self):
        # Full dataset with training transforms
        full_dataset = datasets.ImageFolder(self.DATA_DIR, transform=self.train_transforms)
        targets = [label for _, label in full_dataset.samples]

        # Step 1: Split into train_val (80%) and test (20%)
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.SEED)
        train_val_idx, test_idx = next(sss1.split(np.zeros(len(targets)), targets))
        self.train_val_data = Subset(full_dataset, train_val_idx)
        # Test set (val transforms)
        test_data_base = datasets.ImageFolder(self.DATA_DIR, transform=self.val_test_transforms)
        self.test_dataset = Subset(test_data_base, test_idx)

        # Step 2: Split train_val into train (60%) and val (20% of total)
        targets_train_val = [targets[i] for i in train_val_idx]
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=self.SEED)  # 0.25 of 0.8 = 0.2
        train_idx_in_train_val, val_idx_in_train_val = next(
            sss2.split(np.zeros(len(self.train_val_data)), targets_train_val)
        )
        self.train_dataset = Subset(self.train_val_data, train_idx_in_train_val)
        # Validation set (val transforms)
        val_data_base = datasets.ImageFolder(self.DATA_DIR, transform=self.val_test_transforms)
        original_val_indices = [train_val_idx[i] for i in val_idx_in_train_val]
        self.val_dataset = Subset(val_data_base, original_val_indices)

        # DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True, num_workers=self.NUM_WORKERS)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.BATCH_SIZE, num_workers=self.NUM_WORKERS)

    def _init_model(self):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.NUM_CLASSES)
        self.model = self.model.to(self.DEVICE)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-6)

    def train(self):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_acc = 0.0
        epochs_no_improve = 0

        print(f"Starting training on {self.DEVICE}")

        for epoch in range(self.EPOCHS):
            # ————————— Perform training —————————
            self.model.train()
            total_train_loss = 0.0
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
            avg_train_loss = total_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # ————————— Perform validation —————————
            self.model.eval()
            total_val_loss = 0.0
            correct = total = 0
            with torch.no_grad():
                all_preds, all_labels = [], []
                for inputs, labels in self.val_loader:
                    inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                avg_val_loss = total_val_loss / len(self.val_loader)
                acc = correct / total

            val_losses.append(avg_val_loss)
            print(f"Epoch {epoch+1}/{self.EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {acc*100:.2f}%")

            # ———— Check each metric independently ————
            self.scheduler.step(avg_val_loss)
            improved_loss = avg_val_loss < best_val_loss - self.MIN_DELTA
            improved_acc = acc > best_acc

            if improved_loss:
                best_val_loss = avg_val_loss
            if improved_acc:
                best_acc = acc

            # ———— If *either* improved, save & reset counter ————
            if improved_loss or improved_acc:
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.MODEL_PATH)
                print(f"--> Model saved: "
                      f"Best Val Loss = {best_val_loss:.4f}, "
                      f"Best Val Acc = {best_acc*100:.2f}%")
            else:
                epochs_no_improve += 1
                print(f"Early stopping: {epochs_no_improve}/"
                      f"{self.EARLY_STOPPING_PATIENCE} without improvement.")
                if epochs_no_improve >= self.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after epoch {epoch+1}")
                    break

        self.train_losses = train_losses
        self.val_losses = val_losses

    def evaluate(self):
        print("\n🎯 Final Evaluation on Held-Out Test Set")
        self.model.load_state_dict(torch.load(self.MODEL_PATH))
        self.model.eval()

        test_preds = []
        test_labels = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.DEVICE), labels.to(self.DEVICE)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        # --- Save classification report & F1 ---
        report = classification_report(test_labels, test_preds, target_names=self.CLASS_NAMES, output_dict=True)
        f1 = report['weighted avg']['f1-score']

        # Save classification report and f1 to json
        results = {
            'classification_report': report,
            'weighted_f1': f1
        }
        with open('metrics/final_classification_report.json', 'w') as f:
            json.dump(results, f, indent=4)

        # --- Save confusion matrix as image ---
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=self.CLASS_NAMES, yticklabels=self.CLASS_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig('metrics/final_confusion_matrix.png')
        plt.close()


    def plot_loss(self, save_path="metrics/loss_plot.png"):
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(self.train_losses) + 1), self.train_losses, label="Train Loss")
        plt.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    trainer = PhoneClassifierTrainer()
    trainer.train()
    trainer.evaluate()
    trainer.plot_loss()
