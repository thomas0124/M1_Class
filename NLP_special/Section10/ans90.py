from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "llm-jp/llm-jp-3-150m-instruct3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The movie was full of"

inputs = tokenizer(prompt, return_tensors="pt")
print("トークン化されたプロンプト:")
print(f"トークンID: {inputs['input_ids'][0].tolist()}")
print(
    f"トークン: {[tokenizer.decode([token_id]) for token_id in inputs['input_ids'][0]]}"
)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]
    probabilities = torch.softmax(logits, dim=0)

top_k = 10
top_indices = torch.topk(probabilities, k=top_k).indices
top_probabilities = torch.topk(probabilities, k=top_k).values

print("\n予測されたトークンと確率:")
for i in range(top_k):
    token = tokenizer.decode([top_indices[i]])
    probability = top_probabilities[i].item()
    print(f"{i + 1}. トークン: {token}, 確率: {probability:.4f}")