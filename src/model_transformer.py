# src/model_transformer.py
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "distilbert-base-uncased"

def tokenize_fn(batch, tokenizer):
    return tokenizer(batch["text"], truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

def main():
    ds = load_dataset("yelp_polarity")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized = ds.map(lambda b: tokenize_fn(b, tokenizer), batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # use a subset for a quick demo; remove .select(...) for full training
    small_train = tokenized["train"].shuffle(seed=42).select(range(5000))
    small_test  = tokenized["test"].select(range(2000))

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    args = TrainingArguments(
        output_dir="models/distilbert-yelp",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=small_train,
        eval_dataset=small_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    print(trainer.evaluate())

    # save final model for the app
    trainer.save_model("models/distilbert-yelp/best")
    tokenizer.save_pretrained("models/distilbert-yelp/best")

if __name__ == "__main__":
    main()
