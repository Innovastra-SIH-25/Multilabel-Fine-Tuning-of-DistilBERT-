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
import re

print("🚀 Starting training process...")

# --- 1. Text Cleaning Function ---
def clean_text(examples):
    text = examples["text"]
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[a-zA-Z0-9_]+", "", text)
    text = re.sub(r"#", "", text)
    examples["text"] = text.strip()
    return examples

# --- 2. Load and Prepare Dataset ---
print("📂 Loading dataset...")
df = pd.read_csv("ocean_hazards_tweets.csv")

label_columns = ['tsunami', 'high_waves', 'coastal_flooding', 'not_relevant', 'panic', 'informational', 'help_needed']
df['labels'] = df[label_columns].values.astype(np.float32).tolist()

dataset = Dataset.from_pandas(df[['text', 'labels']])
dataset = dataset.map(clean_text)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"✅ Training samples: {len(train_dataset)}")
print(f"✅ Validation samples: {len(eval_dataset)}")

# --- 3. Calculate Weights for Imbalance ---
print("⚖️ Calculating weights for class imbalance...")
pos_counts = df[label_columns].sum()
neg_counts = len(df) - pos_counts
pos_weights = neg_counts / pos_counts
device = "cuda" if torch.cuda.is_available() else "cpu"
pos_weights_tensor = torch.tensor(pos_weights.values, dtype=torch.float32).to(device)
print("✅ Weights calculated.")

# --- 4. Tokenize Data ---
print("🔤 Tokenizing data...")
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# --- 5. Load Model ---
print("🤖 Loading RoBERTa-base model...")
num_labels = len(label_columns)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    problem_type="multi_label_classification",
    id2label={i: label for i, label in enumerate(label_columns)},
    label2id={label: i for i, label in enumerate(label_columns)}
)

# --- 6. Metrics and Custom Trainer (Corrected) ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[probs >= 0.5] = 1
    
    f1_micro = f1_score(labels, y_pred, average='micro', zero_division=0)
    f1_macro = f1_score(labels, y_pred, average='macro', zero_division=0)
    return {'f1_micro': f1_micro, 'f1_macro': f1_macro}

class WeightedLossTrainer(Trainer):
    # Added **kwargs to accept extra arguments
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# --- 7. Training Arguments ---
training_args = TrainingArguments(
    output_dir="./trained_model",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
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
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=True,
)

# --- 8. Create and Run Trainer ---
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

print("🔥 Starting training with RoBERTa-base...")
trainer.train()

# --- 9. Save and Evaluate ---
print("💾 Saving trained model...")
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
print("✅ Model saved to './trained_model' folder")

print("\n📊 Final evaluation results (at 0.5 threshold):")
results = trainer.evaluate()
print(f"F1 Macro Score: {results['eval_f1_macro']:.4f}")
print(f"F1 Micro Score: {results['eval_f1_micro']:.4f}")

print("\n🔍 Finding optimal prediction threshold...")
preds_output = trainer.predict(tokenized_eval)
labels = preds_output.label_ids
sigmoid = torch.nn.Sigmoid()
probs = sigmoid(torch.Tensor(preds_output.predictions))

best_f1 = 0
best_threshold = 0.5
for threshold in np.arange(0.1, 0.9, 0.05):
    y_pred = np.zeros(probs.shape)
    y_pred[probs >= threshold] = 1
    f1 = f1_score(labels, y_pred, average='macro', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"✅ Best threshold found: {best_threshold:.2f}")
print(f"   - Achieves F1 Macro Score: {best_f1:.4f}")

print("\n🎉 Training completed successfully!")