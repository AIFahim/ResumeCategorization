import pandas as pd
import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder

# Load the dataset (using the cleaned data directly)
train_set = pd.read_csv('data/train_data.csv')
val_set = pd.read_csv('data/val_data.csv')
test_set = pd.read_csv('data/test_data.csv')

# Ensure all entries in 'cleaned_resume' are strings
train_set['cleaned_resume'] = train_set['cleaned_resume'].astype(str).fillna('')
val_set['cleaned_resume'] = val_set['cleaned_resume'].astype(str).fillna('')
test_set['cleaned_resume'] = test_set['cleaned_resume'].astype(str).fillna('')

# Encode labels as integers
label_encoder = LabelEncoder()
train_set['labels'] = label_encoder.fit_transform(train_set['Category'])
val_set['labels'] = label_encoder.transform(val_set['Category'])
test_set['labels'] = label_encoder.transform(test_set['Category'])

# Save the label encoder classes for later use
np.save('label_classes.npy', label_encoder.classes_)

# Convert to Hugging Face dataset format
train_ds = Dataset.from_pandas(train_set)
val_ds = Dataset.from_pandas(val_set)
test_ds = Dataset.from_pandas(test_set)

# Load pre-trained model and tokenizer
model_id = 'ahmedheakl/bert-resume-classification'
tokenizer = BertTokenizer.from_pretrained(model_id)
model = BertForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(label_encoder.classes_),
    ignore_mismatched_sizes=True  # This will handle the size mismatch
)

# Tokenize the dataset
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['cleaned_resume'], padding="max_length", truncation=True)
    tokenized_inputs["labels"] = examples["labels"]
    return tokenized_inputs

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Define the compute metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=12,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate on the test set
eval_results = trainer.evaluate(eval_dataset=test_ds)
print(f"Test Results: {eval_results}")

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_bert_resume_classification")
tokenizer.save_pretrained("./fine_tuned_bert_resume_classification")

# Save the label encoder classes for future inference
np.save("./fine_tuned_bert_resume_classification/label_classes.npy", label_encoder.classes_)
