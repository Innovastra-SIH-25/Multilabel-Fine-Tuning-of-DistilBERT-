# train_model.py
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from datasets import Dataset
import pandas as pd
from sklearn.metrics import f1_score
import numpy as np

print("🚀 Starting training process...")

# 1. Load and prepare the dataset with correct float type
print("📂 Loading dataset...")
df = pd.read_csv("ocean_hazards_tweets.csv")

# Define label columns from your dataset and convert to floats directly
label_columns = ['tsunami', 'high_waves', 'coastal_flooding', 'not_relevant', 'panic', 'informational', 'help_needed']
df['labels'] = df[label_columns].values.astype(np.float32).tolist()

# Convert to dataset and split
dataset = Dataset.from_pandas(df[['text', 'labels']])
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(eval_dataset)}")

# 2. Load tokenizer and tokenize data
print("🔤 Tokenizing data...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 3. Load model for multi-label classification
print("🤖 Loading DistilBERT model...")
num_labels = len(label_columns)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification",
    id2label={i: label for i, label in enumerate(label_columns)},
    label2id={label: i for i, label in enumerate(label_columns)}
)

# 4. Metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[probs >= 0.5] = 1
    
    f1_micro = f1_score(labels, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(labels, y_pred, average='macro', zero_division=0)
    return {'f1_micro': f1_micro, 'f1_macro': f1_macro}

# 5. Training arguments optimized for your system
training_args = TrainingArguments(
    output_dir="./trained_model",
    num_train_epochs=5,
    per_device_train_batch_size=16, # Increased batch size for smaller model
    per_device_eval_batch_size=32, # Increased batch size for smaller model
    learning_rate=2e-5, # Adjusted learning rate
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=300,
    save_strategy="steps",
    save_steps=300,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    report_to="none",
    fp16=True,
    dataloader_pin_memory=True,
)

# 6. Create and run trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

print("🔥 Starting training with DistilBERT...")
print("⏰ This will be much faster than RoBERTa-large! ☕")
trainer.train()

# 7. Save the final model
print("💾 Saving trained model...")
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("✅ Model saved to './trained_model' folder")

# 8. Final evaluation
print("📊 Final evaluation results:")
results = trainer.evaluate()
print(f"F1 Macro Score: {results['eval_f1_macro']:.4f}")
print(f"F1 Micro Score: {results['eval_f1_micro']:.4f}")
print("🎉 Training completed successfully!")