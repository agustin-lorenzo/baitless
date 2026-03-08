import torch
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from transformers import  DataCollatorWithPadding, DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('datasets/comments/all_comments.csv')

# Encode labels, create and tokenize dataset
label_encoder = LabelEncoder()
df['labels'] = label_encoder.fit_transform(df['labels'])
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize(examples):
    return tokenizer(examples['text'], truncation=True)
dataset = Dataset.from_pandas(df).map(tokenize, batched=True).train_test_split(test_size=0.1, shuffle=True)

# Initalize model
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                            num_labels=8,
                                                            id2label=id2label,
                                                            label2id=label2id).to(device)

# Set training metric
metric = evaluate.load('f1')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return metric.compute(predictions=predictions, references=labels, average='weighted')

# Set training arguments
training_args = TrainingArguments(
    num_train_epochs=25,
    output_dir='models/comment_checkpoints',
    learning_rate=1e-5,
    lr_scheduler_type='cosine',
    weight_decay=0.15,
    optim='adamw_torch',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    metric_for_best_model='f1',
    load_best_model_at_end=True,
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

trainer.train()
trainer.save_model('models/comment-model')
