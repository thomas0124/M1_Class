from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math


def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    mean_loss = loss.mean()

    perplexity = math.exp(mean_loss.item())

    return perplexity


model_id = "llm-jp/llm-jp-3-150m-instruct3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

sentences = [
    "The movie was full of surprises",
    "The movies were full of surprises",
    "The movie were full of surprises",
    "The movies was full of surprises",
]

print("各文のパープレキシティ:")
for sentence in sentences:
    perplexity = calculate_perplexity(model, tokenizer, sentence)
    print(f"文: {sentence}")
    print(f"パープレキシティ: {perplexity:.2f}")
    print("-" * 50)