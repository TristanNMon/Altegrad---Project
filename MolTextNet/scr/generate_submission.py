# generate_submission.py
import pandas as pd
from transformers import pipeline

# Load the model you just trained
pipe = pipeline("text2text-generation", model="checkpoints/my_final_model", device=0)

# Load the SMILES you prepared in Step 1
df = pd.read_csv("../data/test_smiles.csv")

print("Generating descriptions...")
descriptions = []

# Process in batches to avoid memory issues
for i, row in df.iterrows():
    if row['smiles'] is None or pd.isna(row['smiles']):
        descriptions.append("Invalid Molecule")
        continue
        
    # Generate text
    out = pipe(row['smiles'], max_length=256, num_beams=5)
    descriptions.append(out[0]['generated_text'])
    
    if i % 100 == 0: print(f"Processed {i} molecules...")

# Create submission file format [cite: 115]
submission = pd.DataFrame({
    "ID": df["id"], # ID from ALTEGRAD data [cite: 39]
    "description": descriptions
})

submission.to_csv("../Output/submission.csv", index=False)
print("Done! Ready to submit.")