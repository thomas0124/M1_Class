from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from datasets import Dataset
import pandas as pd
from trl import DPOTrainer


def create_prompt(sentence):
    return f"""以下の文の感情を分析してください。文の後に「positive」または「negative」のいずれかで答えてください。

文: {sentence}
感情: """


def create_preference_data(df):
    chosen_responses = []
    rejected_responses = []
    prompts = []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        label = row["label"]
        prompt = create_prompt(sentence)
        prompts.append(prompt)
        if label == 1:
            chosen_responses.append(prompt + "positive")
            rejected_responses.append(prompt + "negative")
        else:
            chosen_responses.append(prompt + "negative")
            rejected_responses.append(prompt + "positive")

    return {
        "prompt": prompts,
        "chosen": chosen_responses,
        "rejected": rejected_responses,
    }


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
    train_preference_data = create_preference_data(train_df)
    dev_preference_data = create_preference_data(dev_df)
    train_dataset = Dataset.from_dict(train_preference_data)
    dev_dataset = Dataset.from_dict(dev_preference_data)
    training_args = TrainingArguments(
        output_dir="./results_dpo",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=3,
        logging_steps=1,
        optim="adamw_8bit",
        seed=42,
    )
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
    )
    dpo_trainer.train()
    eval_results = dpo_trainer.evaluate()
    print(f"最終評価結果: {eval_results}")


if __name__ == "__main__":
    main()