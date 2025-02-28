import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型和分词器
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

# 第一次输入
input_text_1 = "我爱中"
input_ids_1 = tokenizer.encode(input_text_1, return_tensors="pt").to("cuda")

# 第一次前向传播，获取 KV Cache
past_key_values = None
outputs_1 = model(input_ids_1, past_key_values=past_key_values)
logits_1 = outputs_1.logits
past_key_values = outputs_1.past_key_values

# 第二次输入
input_text_2 = "我爱中国人民"
input_ids_2 = tokenizer.encode(input_text_2, return_tensors="pt").to("cuda")

# 提取第二次输入中除第一次输入之外的部分
remaining_input_ids = input_ids_2[:, len(input_ids_1[0]):]

# 第二次前向传播，使用第一次的 KV Cache
outputs_2 = model(remaining_input_ids, past_key_values=past_key_values)
logits_2 = outputs_2.logits
new_past_key_values = outputs_2.past_key_values

# 解码生成的文本
generated_ids = input_ids_2[0].tolist()
for _ in range(1):  # 再生成 5 个新 token
    next_token_logits = logits_2[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)
    generated_ids.append(next_token_id.item())

    # 准备下一次输入
    next_input_ids = next_token_id.unsqueeze(-1)
    outputs = model(next_input_ids, past_key_values=new_past_key_values)
    logits_2 = outputs.logits
    new_past_key_values = outputs.past_key_values

generated_text = tokenizer.decode(generated_ids)
print("Generated text:", generated_text)