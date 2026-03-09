import torch
import evaluate
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('datasets/fallacy_binaries.csv')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize(examples):
    return tokenizer(examples['text'], truncation=True)
dataset = Dataset.from_pandas(df).map(tokenize, batched=True).train_test_split(test_size=0.1, shuffle=True)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

metric = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='binary'),
        'precision': precision_score(labels, predictions, average='binary'),
        'recall': recall_score(labels, predictions, average='binary')
    }

# Set training arguments
training_args = TrainingArguments(
    output_dir='models/f-binary_db-checkpoints',
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    num_train_epochs=20,
    weight_decay=0.02,
    optim='adamw_torch',
    eval_strategy='epoch',
    save_strategy='epoch',
    save_total_limit=2,
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy'
)

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,    
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Train and save the model, push to hub
trainer.train()
trainer.save_model('models/f-binary_db-model')
