# predict.py
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Load the trained text classification model
model_path = "./trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create text classification pipeline
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", 
                      model=model, 
                      tokenizer=tokenizer, 
                      device=device,
                      return_all_scores=True)

# Create a separate Named Entity Recognition (NER) pipeline
# We use a pre-trained model for this task
ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", device=device)

# Example tweets to predict
test_tweets = [
    "Massive waves hitting Mumbai beach right now! Very scary situation.",
    "Beautiful sunset at Goa beach today. Perfect weather.",
    "Tsunami warning issued for Chennai coastal areas. Evacuate immediately!",
    "My inbox is flooded with emails today, just like a tsunami.",
    "Water entering homes in coastal areas of Puri. Need help urgently."
]

print("🧪 Making predictions on test tweets:")
print("-" * 50)

for i, tweet in enumerate(test_tweets, 1):
    print(f"\n{i}. Tweet: {tweet}")
    
    # Run the classification model first
    classification_results = classifier(tweet)
    
    # Then run the NER model to get the location
    ner_results = ner_pipeline(tweet)
    
    # Print the classification predictions
    print("   Classification:")
    for label_score in classification_results[0]:
        if label_score['score'] > 0.3:
            print(f"      - {label_score['label']}: {label_score['score']:.3f}")
    
    # Print the extracted locations
    print("   Location:")
    locations_found = False
    for entity in ner_results:
        # Check for location entities
        if entity['entity'].endswith('LOC'):
            print(f"      - Detected Location: {entity['word']} with score {entity['score']:.3f}")
            locations_found = True
    
    if not locations_found:
        print("      - No location found.")