import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch_npu
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

import argparse

from transformers import ViTConfig, ViTModel, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from Qwen3_module import XForCausalLM
from utils import XTrainer

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
        
        # 拼接到最后一个维度：shape -> [B, N, vision_dim + text_dim]
        text_embeds_padded = torch.cat([pad, text_embeds], dim=-1)  # 注意这里调整了拼接顺序
        return text_embeds_padded
    
    def __get_image_embed__(self, pixel_values):        
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
    
    def compute_loss(self, logits, target_answer,):
        """
        计算语言建模损失
        Args:
            logits: 模型输出的logits [batch_size, seq_len, vocab_size]
            target_answer: 目标答案字符串
        """
        _ , _ , vocab_size = logits.shape
        
        # 计算交叉熵损失
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 将logits和labels重塑为2D张量
        flat_logits = logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        flat_labels = target_answer.view(-1)  # [batch_size * seq_len]
        
        loss = criterion(flat_logits, flat_labels)
        return loss
    
    def forward(self, image, text, target_answer=None, max_new_tokens=128):
        self.image, self.text = image, text
        
        text_embed = self.__get_text_embed__(self.text).to(self.device)
        image_embed = self.__get_image_embed__(self.image).to(self.device)
        
        merge_embed = torch.cat([image_embed, text_embed], dim=1).to(self.device)
        attention_mask = torch.ones((merge_embed.shape[0], max_new_tokens + merge_embed.shape[1]), device=self.device, dtype=torch.long)
        
        B = merge_embed.shape[0]
        is_training = target_answer is not None
        
        eos_token_id = getattr(self.config, 'eos_token_id', None)
        
        outputs = self.llm.model(
            inputs_embeds=merge_embed,
            attention_mask=attention_mask[:, :merge_embed.shape[1]],
            use_cache=True,
            return_dict=True
        )
        
        past_key_values = outputs.past_key_values
        current_logits = self.llm.lm_head(outputs.last_hidden_state[:, -1:, :])
        
        all_logits = [] if is_training else None
        generated_ids = torch.zeros((B, max_new_tokens), dtype=torch.long, device=self.device)  # 预分配生成的ID张量
        eos_flags = torch.zeros(B, dtype=torch.bool, device=self.device)  # 跟踪每个样本是否完成生成
        
        for step in range(max_new_tokens):
            step_logits = current_logits.squeeze(1)
            
            if is_training:
                all_logits.append(step_logits)
            
            next_token = step_logits.argmax(-1)
            generated_ids[:, step] = next_token  # 存储生成的token
            
            # 检查EOS条件
            if eos_token_id is not None:
                eos_flags |= (next_token == eos_token_id)
                if eos_flags.all():
                    break
            
            # 更新attention mask
            attention_mask[:, merge_embed.shape[1] + step] = 1
            
            # 只为尚未完成的样本生成下一步
            unfinished_mask = ~eos_flags
            if unfinished_mask.any():
                next_embed = self.llm.model.get_input_embeddings()(next_token[unfinished_mask]).unsqueeze(1)
                outputs = self.llm.model(
                    inputs_embeds=next_embed,
                    attention_mask=attention_mask[unfinished_mask, :merge_embed.shape[1] + step + 1],
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = outputs.past_key_values
                current_logits = self.llm.lm_head(outputs.last_hidden_state[:, -1:, :])
            else:
                break
        
        result = {
            "generated_ids": generated_ids[:, :step + 1],  # 只返回实际生成的部分
        }
        
        if is_training and target_answer is not None:
            target_labels = target_answer[:, :len(all_logits)]
            if target_labels.shape[1] < len(all_logits):
                target_labels = torch.cat([target_labels, torch.full((target_labels.shape[0], len(all_logits) - target_labels.shape[1]), -100, dtype=target_labels.dtype, device=target_labels.device)], dim=1)
            
            stacked_logits = torch.stack(all_logits, dim=1)
            loss = self.compute_loss(stacked_logits, target_labels)
            
            result.update({
                "logits": stacked_logits,
                "loss": loss
            })
        
        return result
        
def run_demo(rank, world_size, args):
    os.environ['RANK'] = str(rank)            
    os.environ['LOCAL_RANK'] = str(rank)      
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = '127.0.0.1'  
    os.environ['MASTER_PORT'] = '6666'
    os.environ["HF_DATASETS_CACHE"] = "/cache/"
    os.environ['TORCHELASTIC_ERROR_FILE'] = '/home/ma-user/work/zym/error.json'

    torch.npu.set_device(rank)  # 分配每个进程对应的设备
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)
    
    model = X_demo()
    trainer = XTrainer(model, args)  
    trainer.train()
    
    dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="NPU Distributed Training")
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per NPU')
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

    