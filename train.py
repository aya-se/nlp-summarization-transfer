# データセットの読み込み
from datasets import load_dataset
data_files = {}
for name in ["train", "val", "test"]:
    data_files[name] = f"data/qmsum/{name}.jsonl" 
data = load_dataset("json", data_files=data_files, cache_dir="data/preloaded")
data["train"][1]["query"]

# モデルの読み込み
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_dir = "facebook/bart-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# データのtokenization
def tokenize(examples):
    inputs = [f"<s>{query}</s>{source}</s>" for query, source in zip(examples["query"], examples["source"])]
    model_inputs = tokenizer(
        inputs,
        max_length=tokenizer.model_max_length,
        add_special_tokens=False,
        truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"], 
            max_length=tokenizer.model_max_length,
            add_special_tokens=False,
            truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_qmsum = data.map(tokenize, batched=True)

# モデルの訓練・ファインチューニング
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="bart-large-qmsum",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    fp16=True,
    push_to_hub=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_qmsum["train"],
    eval_dataset=tokenized_qmsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()