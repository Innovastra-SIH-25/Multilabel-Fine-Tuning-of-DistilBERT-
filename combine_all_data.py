import pandas as pd

def combine_all_data():
    print("🚀 Starting the final data combination process...")

    # --- 1. Load Your Two Datasets ---
    print("📂 Loading your synthetic data and the new disaster response data...")
    synthetic_df = pd.read_csv("ocean_hazards_tweets.csv")
    new_data_df = pd.read_csv("disaster_response_training.csv")

    # --- 2. Process the New Dataset to Match Your Schema ---
    print("✍️  Formatting the new dataset...")
    
    # Define the columns your model needs
    output_columns = ['text', 'tsunami', 'high_waves', 'coastal_flooding', 'not_relevant', 'panic', 'informational', 'help_needed']
    
    # Create an empty DataFrame with the correct structure
    processed_new_df = pd.DataFrame(columns=output_columns)
    
    # Apply mapping rules
    processed_new_df['text'] = new_data_df['message']
    processed_new_df['not_relevant'] = (new_data_df['related'] == 0).astype(int)
    processed_new_df['coastal_flooding'] = new_data_df['floods']
    processed_new_df['high_waves'] = new_data_df['storm']
    processed_new_df['tsunami'] = 0  # No direct mapping
    processed_new_df['help_needed'] = ((new_data_df['request'] == 1) | (new_data_df['aid_related'] == 1)).astype(int)
    processed_new_df['informational'] = ((new_data_df['related'] == 1) & (new_data_df['request'] == 0)).astype(int)
    processed_new_df['panic'] = 0  # No direct mapping

    # --- 3. Combine Your Original Data with the Processed New Data ---
    print("➕ Combining the two datasets into a final version...")
    
    # Ensure your original data only has the necessary columns
    original_data_formatted = synthetic_df[output_columns]
    
    final_df = pd.concat([original_data_formatted, processed_new_df], ignore_index=True)
    
    # Shuffle the combined dataset
    final_df = final_df.sample(frac=1).reset_index(drop=True)

    # --- 4. Save the Final Master Dataset ---
    output_filename = "final_training_dataset.csv"
    final_df.to_csv(output_filename, index=False)
    
    print(f"\n✅ Success! New master dataset created with {len(final_df)} total examples.")
    print(f"   - Saved to '{output_filename}'")

if __name__ == "__main__":
    combine_all_data()