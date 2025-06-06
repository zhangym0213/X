import os
import sys
import warnings
warnings.filterwarnings('ignore')
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'vision_transformer/vit_jax'))

from vision_transformer.vit_jax.models_vit import VisionTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn

model_dir = "./Qwen3-0.6B"

class X_demo(nn.Module):
    def __init__(self):
        super(X_demo, self).__init__() 
        self.__get_model__()

    def __get_model__(self):
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="npu" 
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

X = X_demo()