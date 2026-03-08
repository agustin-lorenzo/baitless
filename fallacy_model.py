import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv('datasets/all_fallacies.csv')

# Encode labels with multi-label binarizer and tokenize dataset
binarizer = MultiLabelBinarizer()
df['labels'] = binarizer.fit_transform(df['labels']).astype(float).tolist()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize(examples):
    return tokenizer(examples['text'], truncation=True)
dataset = Dataset.from_pandas(df).map(tokenize, batched=True)#.train_test_split(test_size=0.1, shuffle=True)

# Initalize model
id2label = {i: label for i, label in enumerate(binarizer.classes_)}
label2id = {label: i for i, label in enumerate(binarizer.classes_)}
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                            num_labels=len(binarizer.classes_),
                                                            problem_type='multi_label_classification',
                                                            id2label=id2label,
                                                            label2id=label2id)\

# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    dist = 1/(1 + np.exp(-logits))
    predictions = (dist > 0.5).astype(int)
    labels = labels.astype(int)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='micro'),
        'precision': precision_score(labels, predictions, average='micro'),
        'recall': recall_score(labels, predictions, average='micro')
    }

# Set training arguments
training_args = TrainingArguments(
    output_dir='models/fallacy-checkpoints',
    learning_rate=2e-5,
    lr_scheduler_type='cosine',
    num_train_epochs=15,
    weight_decay=0.01,
    optim='adamw_torch',
    #eval_strategy='epoch',
    #save_strategy='epoch',
    logging_steps=100,
    #load_best_model_at_end=True,
    metric_for_best_model='f1'
)

trainer = Trainer(
    model = model,
    args=training_args,
    train_dataset=dataset,
    #train_dataset=dataset['train'],
    #eval_dataset=dataset['test'],
    data_collator=DataCollatorWithPadding(tokenizer),
    #compute_metrics=compute_metrics,    
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

# Train and save the model, push to hub
trainer.train()
trainer.save_model('models/fallacy-model')

model.push_to_hub('agustin-lorenzo/fallacy-classifier')
tokenizer.push_to_hub('agustin-lorenzo/fallacy-classifier')
