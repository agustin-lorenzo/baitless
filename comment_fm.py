import torch
import evaluate
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from transformers import DataCollatorWithPadding, DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

