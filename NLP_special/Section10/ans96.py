import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm


def create_prompt(text):
    return f"""以下のテキストの感情を分析してください。ポジティブ（positive）かネガティブ（negative）のどちらかで答えてください。

テキスト: {text}

感情:"""


def predict_sentiment(text, model, tokenizer):
    prompt = create_prompt(text)
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=10,
            do_sample=False,
            temperature=0.0,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = response.lower()
    return 1 if "positive" in response else 0


def main():
    model_id = "llm-jp/llm-jp-3-150m-instruct3"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    dev_path = "../Section7/SST-2/dev.tsv"
    dataset = pd.read_csv(dev_path, sep="\t", header=0)

    correct = 0
    total = 0

    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        text = row["sentence"]
        label = row["label"]

        predicted_label = predict_sentiment(text, model, tokenizer)

        if predicted_label == label:
            correct += 1
        total += 1

    accuracy = correct / total
    print(f"正解率: {accuracy:.4f}")
    print(f"正解数: {correct}")
    print(f"総数: {total}")


if __name__ == "__main__":
    main()