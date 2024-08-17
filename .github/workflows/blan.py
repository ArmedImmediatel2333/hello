from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载预训练模型和分词器
model_name = "gpt2"  # 你也可以尝试更大的模型，如 "gpt2-medium" 或 "gpt2-large"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def generate_response(prompt):
    # 对输入文本进行编码
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # 生成模型的输出
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    
    # 解码生成的文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 测试聊天机器人
if __name__ == "__main__":
    while True:
        user_input = input("你: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        response = generate_response(user_input)
        print("机器人:", response)
