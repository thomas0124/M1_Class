from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "llm-jp/llm-jp-3-150m-instruct3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

chat = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Please answer the following question.",
    },
    {"role": "user", "content": "What do you call a sweet eaten after dinner?"},
]

prompt = tokenizer.apply_chat_template(chat, tokenize=False)
print("生成されたプロンプト:")
print(prompt)
print("\n")

inputs = tokenizer(
    prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
)

outputs = model.generate(
    inputs.input_ids,
    attention_mask=inputs["attention_mask"],
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    repetition_penalty=1.1,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("生成された応答:")
print(response)