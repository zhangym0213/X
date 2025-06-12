import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch_npu
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

import argparse
from tqdm import tqdm

from transformers import ViTConfig, ViTModel, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from Qwen3_module import XForCausalLM
from utils import ImagenetQADataset

model_dir = "/home/ma-user/work/zym/Qwen3-0.6B"

class X_demo(nn.Module):
    def __init__(self, vision_dim=768, vision_ffn=768):
        super(X_demo, self).__init__()
        self.vision_dim = vision_dim
        self.vision_ffn = vision_ffn
        self.device = torch.device("npu:0")

        self.config =  AutoConfig.from_pretrained(model_dir)
        self.config.vision_dim = vision_dim
        self.config.vision_ffn = vision_ffn
        self.__get_model__()

        self.text_dim = self.config.hidden_size
        self.rms_norm_eps = self.config.rms_norm_eps
        self.head_dim = self.config.head_dim
        self.num_attention_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.merged_dim = vision_dim + self.text_dim
        self.bias = self.config.attention_bias
        self.__modify_llm_layers__()
        print(self.vit)  
        print(self.llm)
            
    def __get_model__(self):
        # Qwen3
        self.llm = XForCausalLM.from_pretrained(
            model_dir,
            config=self.config,
            torch_dtype="auto",
            device_map="npu"
        )
        
        # ViT
        self.vit = ViTModel(
            config = ViTConfig(
            hidden_size=self.vision_dim,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3)
        )

    def __modify_llm_layers__(self):
        for layer_idx, layer in enumerate(self.llm.model.layers):       
            # 修改注意力层的q, k, v权重
            attention = layer.self_attn
            
            # 保存原始权重
            old_q_weight = attention.q_proj.weight.data.clone()
            old_k_weight = attention.k_proj.weight.data.clone()
            old_v_weight = attention.v_proj.weight.data.clone()
            old_o_weight = attention.o_proj.weight.data.clone()

            # 创建新的线性层 (从merged_dim到原始输出维度)
            attention.q_proj = nn.Linear(self.merged_dim , old_q_weight.shape[0]+self.vision_dim*2, bias=self.bias).to(old_q_weight.device)
            attention.k_proj = nn.Linear(self.merged_dim , old_k_weight.shape[0]+self.vision_dim, bias=self.bias).to(old_k_weight.device)
            attention.v_proj = nn.Linear(self.merged_dim , old_v_weight.shape[0]+self.vision_dim, bias=self.bias).to(old_v_weight.device)
            attention.o_proj = nn.Linear(old_o_weight.shape[1]+self.vision_dim*2, self.merged_dim , bias=self.bias).to(old_o_weight.device)
            
            # 注意力层中也有Qwen3RMSNorm
            old_q_norm = attention.q_norm.weight.data.clone()
            old_k_norm = attention.k_norm.weight.data.clone()
            attention.q_norm = Qwen3RMSNorm((old_q_weight.shape[0]+self.vision_dim*2)//self.num_attention_heads).to(old_q_norm.device)
            attention.k_norm = Qwen3RMSNorm((old_k_weight.shape[0]+self.vision_dim)//self.num_key_value_heads).to(old_k_norm.device)

            # 初始化新权重 - 将原始权重复制到前text_dim列，其余部分用零初始化
            with torch.no_grad():
                attention.q_proj.weight.data[:old_q_weight.shape[0], :self.text_dim] = old_q_weight    
                attention.k_proj.weight.data[:old_k_weight.shape[0], :self.text_dim] = old_k_weight          
                attention.v_proj.weight.data[:old_v_weight.shape[0], :self.text_dim] = old_v_weight
                attention.o_proj.weight.data[:self.text_dim, :old_o_weight.shape[1]] = old_o_weight
                attention.q_norm.weight.data[:old_q_norm.shape[0]] = old_q_norm
                attention.k_norm.weight.data[:old_k_norm.shape[0]] = old_k_norm
            
            # 修改RMSNorm层
            old_input_norm_weight = layer.input_layernorm.weight.data.clone()
            layer.input_layernorm = Qwen3RMSNorm(self.merged_dim).to(old_input_norm_weight.device)
            
            with torch.no_grad():
                # 将原始权重扩展到新维度，新增部分初始化为1
                layer.input_layernorm.weight.data[:self.text_dim] = old_input_norm_weight
                layer.input_layernorm.weight.data[self.text_dim:] = 1.0
            
            # 后归一化
            old_post_norm_weight = layer.post_attention_layernorm.weight.data.clone()
            layer.post_attention_layernorm = Qwen3RMSNorm(self.merged_dim).to(old_post_norm_weight.device)
            
            with torch.no_grad():
                layer.post_attention_layernorm.weight.data[:self.text_dim] = old_post_norm_weight
                layer.post_attention_layernorm.weight.data[self.text_dim:] = 1.0
            
            # 修改FFN层
            mlp = layer.mlp
            
            # 保存原始权重
            old_gate_weight = mlp.gate_proj.weight.data.clone()
            old_up_weight = mlp.up_proj.weight.data.clone()
            old_down_weight = mlp.down_proj.weight.data.clone()
            
            # 修改gate_proj和up_proj (输入维度从text_dim变为merged_dim)
            mlp.gate_proj = nn.Linear(self.merged_dim, old_gate_weight.shape[0]+self.vision_ffn, bias=self.bias).to(old_gate_weight.device)
            mlp.up_proj = nn.Linear(self.merged_dim, old_up_weight.shape[0]+self.vision_ffn, bias=self.bias).to(old_up_weight.device)
            mlp.down_proj = nn.Linear(old_down_weight.shape[1]+self.vision_ffn, self.merged_dim, bias=self.bias).to(old_down_weight.device)
            
            with torch.no_grad():
                mlp.gate_proj.weight.data[ :old_gate_weight.shape[0], :self.text_dim] = old_gate_weight  
                mlp.up_proj.weight.data[:old_up_weight.shape[0], :self.text_dim] = old_up_weight
                mlp.down_proj.weight.data[:self.text_dim, :old_down_weight.shape[1]] = old_down_weight
        
        # 修改最终的norm层
        old_final_norm_weight = self.llm.model.norm.weight.data.clone()
        self.llm.model.norm = Qwen3RMSNorm(self.merged_dim).to(old_final_norm_weight.device)
        old_lm_head_weight = self.llm.lm_head.weight.data.clone()
        self.llm.lm_head = nn.Linear(self.merged_dim, old_lm_head_weight.shape[0], bias=self.bias).to(old_lm_head_weight.device)

        with torch.no_grad():
            self.llm.model.norm.weight.data[:self.text_dim] = old_final_norm_weight
            self.llm.model.norm.weight.data[self.text_dim:] = 1.0
            self.llm.lm_head.weight.data[:,:self.text_dim] = old_lm_head_weight

    def __get_text_embed__(self, input_ids):
        text_embeds = self.llm.model.embed_tokens(input_ids)
        
        # 创建零向量用于拼接
        pad = torch.zeros(
            text_embeds.size(0),
            text_embeds.size(1),
            self.vision_dim,
            dtype=text_embeds.dtype,
            device=text_embeds.device
        )
        
        print(text_embeds.shape)
        print(pad.shape)
        # 拼接到最后一个维度：shape -> [B, N, vision_dim + text_dim]
        text_embeds_padded = torch.cat([pad, text_embeds], dim=-1)  # 注意这里调整了拼接顺序
        return text_embeds_padded
    
    def __get_patch_embed__(self, pixel_values):        
        # 去掉一个[CLS]，[Batch Size, Patch Num, Vision Dim]
        image_embeds = self.vit.embeddings(pixel_values)[:, 1:, :]

        # 创建零向量用于拼接
        pad = torch.zeros(
            image_embeds.size(0),
            image_embeds.size(1),
            self.text_dim,
            dtype=image_embeds.dtype,
            device=image_embeds.device
        )

        # 拼接到最后一个维度：shape -> [B, N, vision_dim + text_dim]
        image_embeds_padded = torch.cat([image_embeds, pad], dim=-1)  # 注意这里调整了拼接顺序
        return image_embeds_padded
    
    def compute_loss(self, logits, target_answer, input_length):
        """
        计算语言建模损失
        Args:
            logits: 模型输出的logits [batch_size, seq_len, vocab_size]
            target_answer: 目标答案字符串
            input_length: 输入序列长度(用于确定答案开始位置)
        """
        # 将目标答案转换为token ids
        target_encoding = self.tokenizer(
            target_answer, 
            return_tensors="pt", 
            add_special_tokens=False  # 不添加特殊token，因为我们只要答案部分
        )
        target_ids = target_encoding["input_ids"].to(logits.device)  # [1, answer_length]
        
        # 构建完整的目标序列
        # 前面是输入部分(图像+问题)，后面是答案部分
        batch_size, total_seq_len, vocab_size = logits.shape
        
        # 创建标签张量，初始化为-100(ignore_index)
        labels = torch.full((batch_size, total_seq_len), -100, dtype=torch.long, device=logits.device)
        
        # 只在答案部分设置真实标签
        answer_start = input_length  # 答案从输入结束后开始
        answer_length = target_ids.shape[1]
        
        if answer_start + answer_length <= total_seq_len:
            # 如果答案能完全放入序列中
            labels[0, answer_start:answer_start + answer_length] = target_ids[0]
        else:
            # 如果答案超出序列长度，截断答案
            available_length = total_seq_len - answer_start
            if available_length > 0:
                labels[0, answer_start:] = target_ids[0, :available_length]
        
        # 计算交叉熵损失
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 将logits和labels重塑为2D张量
        shift_logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        shift_labels = labels.view(-1)  # [batch_size * seq_len]
        
        loss = loss_fct(shift_logits, shift_labels)
        return loss
    
    def forward(self, image, text, target_answer=None):
        self.image, self.text = image, text
        
        # 获取图像和文本嵌入(input_ids & pixel_values)
        text_embed = self.__get_text_embed__(self.text).to(self.device)
        image_embed = self.__get_image_embed__(self.image).to(self.device)

        # 合并嵌入
        merge_embed = torch.cat([image_embed, text_embed], dim=1).to(self.device)
        
        # 将merge_embed送入LLM的decoder
        # 注意：由于我们修改了模型结构，直接使用generate可能需要额外处理
        # 这里展示如何通过模型的forward传递
        
        # 创建attention mask
        attention_mask = torch.ones(merge_embed.shape[:2], device=self.device, dtype=torch.long)
        
        # 通过LLM处理
        outputs = self.llm.model(
            inputs_embeds=merge_embed,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取最后的hidden states
        hidden_states = outputs.last_hidden_state
        
        # 通过language model head得到logits
        logits = self.llm.lm_head(hidden_states)
        
        if target_answer is not None:
            loss = self.compute_loss(logits, target_answer, merge_embed.shape[1])
        
        return {"logits": logits, "loss": loss}
    
  

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

def run_demo(rank, world_size, args):
    os.environ['RANK'] = str(rank)            
    os.environ['LOCAL_RANK'] = str(rank)      
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'  
    os.environ['MASTER_PORT'] = '6666'
    os.environ["HF_DATASETS_CACHE"] = "/cache/"

    torch.npu.set_device(rank)  # 分配每个进程对应的设备
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    
    model = X_demo()
    trainer = XTrainer(model, args)  
    trainer.train()
    
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="NPU Distributed Training")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size per NPU')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Where to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard log dir')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank passed by torchrun')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()    
    torch.npu.set_device(args.local_rank)
    world_size = torch.npu.device_count()
    mp.spawn(run_demo, args=(world_size, args), nprocs=world_size, join=True)

    