import pandas as pd

print("🔍 Checking dataset balance...")
df = pd.read_csv('final_training_dataset.csv')

print("Total tweets:", len(df))
print("\n--- Hazard Distribution ---")
print("Tsunami tweets:", df['tsunami'].sum())
print("High Waves tweets:", df['high_waves'].sum())
print("Coastal Flooding tweets:", df['coastal_flooding'].sum())
print("Not Relevant tweets:", df['not_relevant'].sum())

print("\n--- Sentiment Distribution (within Hazards) ---")
hazard_df = df[df['not_relevant'] == 0]
print("Panic tweets:", hazard_df['panic'].sum())
print("Informational tweets:", hazard_df['informational'].sum())
print("Help Needed tweets:", hazard_df['help_needed'].sum())