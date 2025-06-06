import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
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

    def __get_patch_embed__(self, image):
        inputs = self.image_processor(images=image, return_tensors="pt")
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

    def __get_text_embed__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
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
        
        # 获取图像和文本嵌入
        image_embed = self.__get_patch_embed__(image)
        text_embed = self.__get_text_embed__(text)
        
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
        
        if target_answer is not None:
            loss = self.compute_loss(logits, target_answer, merge_embed.shape[1])
        
        return {"logits": logits, "loss": loss}

if __name__ == "__main__":
    model = X_demo(vision_dim=768, text_dim=1024)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # 下载并加载 CIFAR-100 训练集
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

    # 下载并加载 CIFAR-100 测试集
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
    
    cifar100_classes = trainset.classes
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000)
    
    max_epoch = 10

    for epoch in tqdm(range(max_epoch)):
        model.train()
        total_loss = 0
        for images, labels in trainloader:
            outputs = model(images, "What is in the image?", [cifar100_classes[label] for label in labels])
            loss = outputs["loss"]
            total_loss += loss.item()

            # 反向传播
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度

            # 梯度裁剪(可选，防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # 更新参数

            # 更新学习率
            if scheduler is not None:
                scheduler.step()
        
        print(f"Epoch {epoch + 1}/{max_epoch}, Train Loss: {total_loss / len(trainloader)}")
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images, "What is in the image?", [cifar100_classes[label] for label in labels])
                logits = outputs["logits"]
                _, predicted = torch.max(logits, dim=2)
                total += labels.size(0)
                correct += (predicted[:, -1] == labels).sum().item()
        
        print(f"Test Accuracy: {100 * correct / total:.2f}%")
        
        # 展示 10 个例子的模型输出文本
        with torch.no_grad():
            for images, labels in testloader:
                outputs = model(images, "What is in the image?")
                logits = outputs["logits"]
                _, predicted = torch.max(logits, dim=2)
                for i in range(min(10, len(images))):
                    print(f"Image {i + 1}: Predicted: {cifar100_classes[predicted[i, -1]]}, Actual: {cifar100_classes[labels[i]]}")
                break