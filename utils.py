import os
from PIL import Image
from torch.utils.data import Dataset
from classes import IMAGENET2012_CLASSES
from transformers import AutoTokenizer, ViTImageProcessor

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

        label_encoding = self.tokenizer(label, return_tensors="pt", padding="max_length", max_length=64)
        label_ids = label_encoding.input_ids.squeeze(0)
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return (self.image_processor(images=Image.open(path).convert('RGB'), return_tensors="pt").pixel_values,
                self.tokenizer(self.prompt, return_tensors="pt", padding=True).input_ids.squeeze(),
                label_ids)
    
    def __get_model__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        self.image_processor = ViTImageProcessor(do_resize=True, size={"height": 224, "width": 224}, do_normalize=True)
