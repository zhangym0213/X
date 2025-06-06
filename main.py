import os
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from PIL import Image
from transformers import ViTConfig, ViTModel, ViTImageProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "./Qwen3-0.6B"

class X_demo(nn.Module):
    def __init__(self, vision_dim=768, text_dim=1024):
        super(X_demo, self).__init__()
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.merged_dim = vision_dim + text_dim  
        self.__get_model__()
        self.__modify_llm_layers__()  
        
    def __get_model__(self):
        # Qwen3
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            device_map="npu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
        
        # ViT
        self.image_processor = ViTImageProcessor(
            do_resize=True,
            size={"height": 224, "width": 224},
            do_normalize=True
        )
        config = ViTConfig(
            hidden_size=self.vision_dim,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=224,
            patch_size=16,
            num_channels=3
        )
        self.vit = ViTModel(config)
    
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
            attention.q_proj = nn.Linear(self.merged_dim, old_q_weight.shape[0]+self.vision_dim, bias=False).to(old_q_weight.device)
            attention.k_proj = nn.Linear(self.merged_dim, old_k_weight.shape[0]+self.vision_dim, bias=False).to(old_k_weight.device)
            attention.v_proj = nn.Linear(self.merged_dim, old_v_weight.shape[0]+self.vision_dim, bias=False).to(old_v_weight.device)
            
            # 初始化新权重 - 将原始权重复制到前text_dim列，其余部分用零初始化
            with torch.no_grad():
                attention.q_proj.weight.data[:self.text_dim, :old_q_weight.shape[1]] = old_q_weight    
                attention.k_proj.weight.data[:self.text_dim, :old_k_weight.shape[1]] = old_k_weight          
                attention.v_proj.weight.data[:self.text_dim, :old_v_weight.shape[1]] = old_v_weight

            
            # 修改RMSNorm层
            # 输入归一化
            old_input_norm_weight = layer.input_layernorm.weight.data.clone()
            layer.input_layernorm = nn.RMSNorm(self.merged_dim, eps=layer.input_layernorm.eps).to(old_input_norm_weight.device)
            
            with torch.no_grad():
                # 将原始权重扩展到新维度，新增部分初始化为1
                layer.input_layernorm.weight.data[:self.text_dim] = old_input_norm_weight
                layer.input_layernorm.weight.data[self.text_dim:] = 1.0
            
            # 后归一化
            old_post_norm_weight = layer.post_attention_layernorm.weight.data.clone()
            layer.post_attention_layernorm = nn.RMSNorm(self.merged_dim, eps=layer.post_attention_layernorm.eps).to(old_post_norm_weight.device)
            
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
            mlp.gate_proj = nn.Linear(self.merged_dim, old_gate_weight.shape[0], bias=False).to(old_gate_weight.device)
            mlp.up_proj = nn.Linear(self.merged_dim, old_up_weight.shape[0], bias=False).to(old_up_weight.device)
            
            with torch.no_grad():
                mlp.gate_proj.weight.data[:self.text_dim, :old_gate_weight.shape[1]] = old_gate_weight  
                mlp.up_proj.weight.data[:self.text_dim, :old_up_weight.shape[1]] = old_up_weight
        
        # 修改最终的norm层
        old_final_norm_weight = self.llm.model.norm.weight.data.clone()
        self.llm.model.norm = nn.RMSNorm(self.merged_dim, eps=self.llm.model.norm.eps).to(old_final_norm_weight.device)
        
        with torch.no_grad():
            self.llm.model.norm.weight.data[:self.text_dim] = old_final_norm_weight
            self.llm.model.norm.weight.data[self.text_dim:] = 1.0
        

    def __get_patch_embed__(self):
        inputs = self.image_processor(images=self.image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.vit.device)
        
        # 去掉一个[CLS]，[Batch Size, Patch Num, Vision Dim]
        patch_embeds = self.vit.embeddings(pixel_values)[:, 1:, :]
        
        # 创建零向量用于拼接
        pad = torch.zeros(
            patch_embeds.size(0),
            patch_embeds.size(1),
            self.text_dim,
            dtype=patch_embeds.dtype,
            device=patch_embeds.device
        )
        
        # 拼接到最后一个维度：shape -> [B, N, vision_dim + text_dim]
        patch_embeds_padded = torch.cat([patch_embeds, pad], dim=-1)
        return patch_embeds_padded

    def __get_text_embed__(self):
        inputs = self.tokenizer(self.text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.llm.device)
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

    def forward(self, image):
        self.image, self.text = image, "What is in the image?"
        
        # 获取图像和文本嵌入
        image_embed = self.__get_patch_embed__()
        text_embed = self.__get_text_embed__()
        
        # 合并嵌入
        merge_embed = torch.cat([image_embed, text_embed], dim=1)
        
        # 将merge_embed送入LLM的decoder
        # 注意：由于我们修改了模型结构，直接使用generate可能需要额外处理
        # 这里展示如何通过模型的forward传递
        
        # 创建attention mask
        attention_mask = torch.ones(merge_embed.shape[:2], device=merge_embed.device, dtype=torch.long)
        
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
        
        return logits


if __name__ == "__main__":
    model = X_demo(vision_dim=768, text_dim=1024)
    image = Image.open("your_image.jpg")  # 替换为您的图像路径
    
    # 前向传播
    with torch.no_grad():
        logits = model(image)
        predicted_ids = torch.argmax(logits, dim=-1)
        
        response = model.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
        print(f"生成的响应: {response}")