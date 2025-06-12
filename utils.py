import os
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from classes import IMAGENET2012_CLASSES
from transformers import AutoTokenizer, ViTImageProcessor

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

model_dir = "/home/ma-user/work/zym/Qwen3-0.6B"

class ImagenetQADataset(Dataset):
    def __init__(self, root_dir = "/cache/imagenet-1k/data/", split="train"):
        self.__get_model__()
        self.root_dir = os.path.join(root_dir, split)
        self.prompt = "What is in the image?"
        self.samples = []
        
        for fname in os.listdir(self.root_dir):
            if fname.endswith(".JPEG"):
                parts = fname.split("_")
                synset = parts[-1].split(".")[0]
                if synset in IMAGENET2012_CLASSES:
                    label = IMAGENET2012_CLASSES[synset]
                    self.samples.append((os.path.join(self.root_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        label_encoding = self.tokenizer(label, return_tensors="pt", padding="max_length", max_length=128)
        label_ids = label_encoding.input_ids.squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return (self.image_processor(images=Image.open(path).convert('RGB'), return_tensors="pt").pixel_values.squeeze(),
                self.tokenizer(self.prompt, return_tensors="pt", padding=True).input_ids.squeeze(),
                label_ids)
    
    def __get_model__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.image_processor = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224}, do_normalize=True)

class XTrainer:
    def __init__(self, model, args):
        self.model = model
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.device = torch.device("npu")
        self.log_dir = args.log_dir

        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        self.train_dataset = ImagenetQADataset(split="train")
        self.val_dataset = ImagenetQADataset(split= "val")

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.train_sampler, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=self.val_sampler, num_workers=4)

        self.model = DDP(self.model.to(self.device), device_ids=[self.rank])

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs * len(self.train_loader))
        self.writer = SummaryWriter(self.log_dir) if self.rank == 0 else None

    def compute_accuracy(self, logits, labels):
        pred_ids = torch.argmax(logits, dim=-1)
        correct = 0
        total = 0
        for pred, target in zip(pred_ids, labels):
            pred_text = self.tokenizer.decode(pred, skip_special_tokens=True)
            target_text = self.tokenizer.decode(target, skip_special_tokens=True)
            if pred_text.strip().lower() in target_text.strip().lower():
                correct += 1
            total += 1
        return correct / total

    def train(self):
        global_step = 0
        best_val_loss = float('inf') 
        for epoch in range(self.epochs):
            self.model.train()
            self.train_sampler.set_epoch(epoch)
            running_loss = 0.0
            for step, (img, text, label) in tqdm(enumerate(self.train_loader)):
                outputs = self.model(img, text, label)
                loss = outputs["loss"]
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                running_loss += loss.item()

                if step % 100 == 0 and self.rank == 0:
                    avg_loss = running_loss / (step + 1)
                    print(f"Epoch {epoch} Step {step}: Loss = {avg_loss:.4f}")
                    self.writer.add_scalar("train/loss", avg_loss, global_step)
                global_step += 1

            if self.rank == 0:
                val_loss, val_acc = self.evaluate()
                print(f"Epoch {epoch} finished. Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
                self.writer.add_scalar("val/loss", val_loss, epoch)
                self.writer.add_scalar("val/accuracy", val_acc, epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(self.output_dir, "best_checkpoint.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'accuracy': val_acc,
                    }, checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path} with val_loss: {val_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, texts, labels in tqdm(self.val_loader):
                answers = [self.val_dataset.dataset.features["label"].int2str(label.item()) for label in labels]
                tokenized = self.tokenizer(answers, return_tensors="pt", padding=True).input_ids.to(self.device)

                outputs = self.model(images, texts, tokenized)
                total_loss += outputs["loss"].item()
                logits = outputs["logits"]
                acc = self.compute_accuracy(logits, tokenized)
                correct += acc * len(images)
                total += len(images)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy 