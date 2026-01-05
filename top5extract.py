import os
import pandas as pd
import shutil

# ============================
# User configuration (EDIT)
# ============================

CSV_PATH = r"C:\fungidata\FungiTastic\dataset\FungiTastic\metadata\FungiTastic-Mini\FungiTastic-Mini-Train.csv"                    # path to your CSV file
SOURCE_FOLDER = r"C:\fungidata\FungiTastic\dataset\FungiTastic\FungiTastic-Mini\train\300p"                  # folder containing all original files
OUTPUT_FOLDER = "top5_species_files/"         # folder where extracted files will go
OUTPUT_CSV = "top5_species_data.csv"          # output CSV with filtered rows

FILENAME_COLUMN = "filename"  # column with filenames
SPECIES_COLUMN = "species"    # column with species names

# ============================
# Script starts here
# ============================

# Load CSV
df = pd.read_csv(CSV_PATH)

# Find top 5 most frequent species
top5_species = df[SPECIES_COLUMN].value_counts().head(5).index.tolist()
print("\n📌 Top 5 species found:\n", top5_species, "\n")

# Filter rows only for top 5 species
filtered_df = df[df[SPECIES_COLUMN].isin(top5_species)]

# Save filtered CSV
filtered_df.to_csv(OUTPUT_CSV, index=False)
print(f"💾 New CSV saved as: {OUTPUT_CSV} ({len(filtered_df)} rows)\n")

# Extract unique filenames for these species
filenames = filtered_df[FILENAME_COLUMN].dropna().unique()

# Create output folder
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Copy files
copied = 0
for fname in filenames:
    src = os.path.join(SOURCE_FOLDER, fname)
    dst = os.path.join(OUTPUT_FOLDER, fname)

    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1
    else:
        print(f"⚠️ File not found in source: {fname}")

print(f"✅ Done! Copied {copied} files to '{OUTPUT_FOLDER}'.\n")
