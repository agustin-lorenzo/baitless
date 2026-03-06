import torch
import evaluate
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from transformers import DataCollatorWithPadding, DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and tokenzie dataset
dataset = load_dataset("tasksource/logical-fallacy")
dataset = dataset.rename_column("source_article", "text")
dataset = dataset.rename_column("logical_fallacies", "labels")

# Encode labels
le = LabelEncoder()
all_labels = list(dataset["train"]["labels"]) + list(dataset["test"]["labels"])
le.fit(all_labels)
id2label = {i: label for i, label in enumerate(le.classes_)}
label2id = {label: i for i, label in enumerate(le.classes_)}
def encode_labels(examples):
    examples["labels"] = le.transform(examples["labels"]).tolist()
    return examples
dataset = dataset.map(encode_labels, batched=True)

# Tokenize the inputs
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True)
dataset = dataset.map(tokenize, batched=True)

# Define model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id)
model.to(device)

# Define metric for model training
metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

# Set training arguments
training_args = TrainingArguments(
    output_dir='checkpoints',
    learning_rate=1e-5,
    num_train_epochs=25,
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    optim="adamw_torch",
    weight_decay=0.1,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
    data_collator=DataCollatorWithPadding(tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Train and save the model
trainer.train()
trainer.save_model("fallacy-model")
