import torch
from sklearn import preprocessing
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, pipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset("tasksource/logical-fallacy")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(examples):
    return tokenizer(examples['source_article'], padding='max_length', truncation=True)
dataset = dataset.map(tokenize, batched=True)

print(dataset)

# Encode lables
# label_encoder = preprocessing.LabelEncoder()
# ds['logical_fallacies'] = label_encoder.fit_transform(ds['logical_fallacies'])
# print(ds['logical_fallacies'])

# # Load model and tokenizer
# id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
# label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
# tokenizer = DistilbertTokenizer.from_pretrained('distilbert-base-uncased')
