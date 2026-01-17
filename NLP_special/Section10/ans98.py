from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
from sklearn.metrics import accuracy_score


def create_prompt(sentence):
    return f"""以下の文の感情を分析してください。文の後に「positive」または「negative」のいずれかで答えてください。

文: {sentence}
感情: """


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    predicted_labels = []
    for pred in predictions:
        text = tokenizer.decode(pred, skip_special_tokens=True)
        if "positive" in text.lower():
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    return {"accuracy": accuracy_score(labels, predicted_labels)}


def main():
    model_name = "llm-jp/llm-jp-3-150m-instruct3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id

    train_path = "../Section7/SST-2/train.tsv"
    dev_path = "../Section7/SST-2/dev.tsv"

    train_df = pd.read_csv(train_path, sep="\t", header=0)
    dev_df = pd.read_csv(dev_path, sep="\t", header=0)

    def tokenize_function(examples):
        prompts = [create_prompt(s) for s in examples["sentence"]]
        labels = [
            "positive" if label == 1 else "negative" for label in examples["label"]
        ]
        texts = [prompt + label for prompt, label in zip(prompts, labels)]
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(dev_df)

    train_dataset = train_dataset.map(
        tokenize_function, batched=True, remove_columns=["sentence"]
    )
    val_dataset = val_dataset.map(
        tokenize_function, batched=True, remove_columns=["sentence"]
    )

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_dir="./logs",
        save_strategy="epoch",
        eval_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"最終評価結果: {eval_results}")


if __name__ == "__main__":
    main()