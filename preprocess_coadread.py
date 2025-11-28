"""
COADREAD Data Preprocessing Script
Adapts the BRCA preprocessing pipeline for Colorectal Cancer data

Compatible with BOTH:
- coadread_tcga (original TCGA, ~276 samples) ← RECOMMENDED
- coadread_tcga_pan_can_atlas_2018 (PanCancer Atlas, ~600 samples)

Requirements:
1. Download COADREAD data from cBioPortal:
   https://www.cbioportal.org/study/summary?id=coadread_tcga
2. Place files in 'coadread_raw/' folder:
   - data_mutations.txt (mutation data)
   - data_cna.txt (copy number alterations)
   - data_clinical_patient.txt (clinical/subtype data)

Output:
- COADREAD_training_data.txt
- COADREAD_validation_data.txt
- COADREAD_training_labels.txt
- COADREAD_validation_labels.txt
- COADREAD_edge2features.txt (if network data available)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ==================================================
# CONFIGURATION
# ==================================================
INPUT_DIR = 'coadread_raw/'
OUTPUT_DIR = 'coadread_processed/'
TRAIN_SPLIT = 0.67  # 67% training, 33% validation (same as BRCA)

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("COADREAD Data Preprocessing")
print("="*60)

# ==================================================
# STEP 1: Load Cancer Gene Lists (optional filtering)
# ==================================================
print("\nStep 1: Loading cancer gene lists...")

# You can use the same cancer gene lists as BRCA
# Or skip this step and use all genes
try:
    cancergenes = set()
    
    # Try to load from BRCA data directory if available
    if os.path.exists('data/CGs.txt'):
        with open('data/CGs.txt') as f:
            for line in f.read().rstrip().splitlines():
                cancergenes.add(line)
        print(f"  ✓ Loaded {len(cancergenes)} cancer genes")
    else:
        print("  ! Cancer gene list not found - will use all genes")
        cancergenes = None
except Exception as e:
    print(f"  ! Warning: {e}")
    cancergenes = None

# ==================================================
# STEP 2: Parse Mutation Data
# ==================================================
print("\nStep 2: Parsing mutation data...")

try:
    # Load mutation data
    df_mut = pd.read_csv(INPUT_DIR + 'data_mutations.txt', sep='\t', comment='#')
    print(f"  ✓ Loaded {len(df_mut)} mutation records")
    
    # Extract patient ID and gene name
    # TCGA patient IDs are first 12 characters: TCGA-XX-XXXX
    df_mut['patient_id'] = df_mut['Tumor_Sample_Barcode'].str[:12]
    
    # Filter for non-silent mutations
    silent_types = ['Silent', '3\'UTR', '5\'UTR', 'Intron', 'IGR']
    df_mut = df_mut[~df_mut['Variant_Classification'].isin(silent_types)]
    print(f"  ✓ After filtering silent mutations: {len(df_mut)} records")
    
    # Optional: Filter for cancer genes only
    if cancergenes is not None:
        df_mut = df_mut[df_mut['Hugo_Symbol'].isin(cancergenes)]
        print(f"  ✓ After filtering cancer genes: {len(df_mut)} records")
    
    # Create binary mutation matrix: patients x genes
    # Value = 1 if gene is mutated in patient, 0 otherwise
    df_mut_matrix = df_mut.groupby(['patient_id', 'Hugo_Symbol']).size().unstack(fill_value=0)
    df_mut_matrix = (df_mut_matrix > 0).astype(int)
    
    print(f"  ✓ Mutation matrix: {df_mut_matrix.shape[0]} patients × {df_mut_matrix.shape[1]} genes")
    
except Exception as e:
    print(f"  ✗ Error parsing mutations: {e}")
    print("  ! Make sure 'data_mutations.txt' is in 'coadread_raw/' folder")
    exit(1)

# ==================================================
# STEP 3: Parse Copy Number Alterations (CNA)
# ==================================================
print("\nStep 3: Parsing copy number alterations...")

try:
    df_cna = pd.read_csv(INPUT_DIR + 'data_cna.txt', sep='\t', index_col=0)
    print(f"  ✓ Loaded CNA data: {df_cna.shape[0]} genes × {df_cna.shape[1]} samples")
    
    # Convert sample IDs to patient IDs (first 12 characters)
    df_cna.columns = df_cna.columns.str[:12]
    
    # Filter for cancer genes if available
    if cancergenes is not None:
        df_cna = df_cna[df_cna.index.isin(cancergenes)]
    
    # Convert CNA values:
    # -2 = deep deletion (homozygous deletion)
    # -1 = shallow deletion
    #  0 = diploid
    # +1 = low-level gain
    # +2 = high-level amplification
    
    # Binary encoding: 1 if amplification (+2) or deletion (-2), else 0
    df_cna_binary = ((df_cna == 2) | (df_cna == -2)).astype(int)
    df_cna_binary = df_cna_binary.T  # Transpose to patients x genes
    
    print(f"  ✓ CNA matrix: {df_cna_binary.shape[0]} patients × {df_cna_binary.shape[1]} genes")
    
except Exception as e:
    print(f"  ! Warning: Could not parse CNA data: {e}")
    print("  ! Continuing with mutations only...")
    df_cna_binary = None

# ==================================================
# STEP 4: Combine Mutations and CNAs
# ==================================================
print("\nStep 4: Combining mutation and CNA data...")

if df_cna_binary is not None:
    # Combine: patient is "altered" if mutation OR significant CNA
    all_genes = set(df_mut_matrix.columns) | set(df_cna_binary.columns)
    all_patients = set(df_mut_matrix.index) | set(df_cna_binary.index)
    
    # Create combined matrix
    df_combined = pd.DataFrame(0, index=sorted(all_patients), columns=sorted(all_genes))
    
    # Add mutations
    for patient in df_mut_matrix.index:
        for gene in df_mut_matrix.columns:
            if df_mut_matrix.loc[patient, gene] == 1:
                df_combined.loc[patient, gene] = 1
    
    # Add CNAs
    for patient in df_cna_binary.index:
        for gene in df_cna_binary.columns:
            if gene in df_combined.columns and df_cna_binary.loc[patient, gene] == 1:
                df_combined.loc[patient, gene] = 1
    
    print(f"  ✓ Combined matrix: {df_combined.shape[0]} patients × {df_combined.shape[1]} genes")
else:
    df_combined = df_mut_matrix
    print(f"  ✓ Using mutations only: {df_combined.shape[0]} patients × {df_combined.shape[1]} genes")

# ==================================================
# STEP 5: Parse Clinical/Subtype Labels
# ==================================================
print("\nStep 5: Parsing subtype labels...")

try:
    df_clinical = pd.read_csv(INPUT_DIR + 'data_clinical_patient.txt', sep='\t', comment='#')
    print(f"  ✓ Loaded clinical data for {len(df_clinical)} patients")
    
    # Look for subtype information
    # Common column names: 'SUBTYPE', 'MSI_STATUS', 'CMS_LABEL', etc.
    subtype_columns = [col for col in df_clinical.columns if 'SUBTYPE' in col.upper() or 
                       'MSI' in col.upper() or 'CMS' in col.upper()]
    
    print(f"  ✓ Found subtype columns: {subtype_columns}")
    
    # Use MSI status if available (MSI-High vs MSS/MSI-Low)
    if any('MSI' in col.upper() for col in df_clinical.columns):
        msi_col = [col for col in df_clinical.columns if 'MSI' in col.upper()][0]
        df_clinical['SUBTYPE'] = df_clinical[msi_col]
        print(f"  ✓ Using MSI status as subtype")
    # Or use CMS subtypes (CMS1, CMS2, CMS3, CMS4)
    elif any('CMS' in col.upper() for col in df_clinical.columns):
        cms_col = [col for col in df_clinical.columns if 'CMS' in col.upper()][0]
        df_clinical['SUBTYPE'] = df_clinical[cms_col]
        print(f"  ✓ Using CMS classification as subtype")
    else:
        # Create binary subtype based on stage
        if 'AJCC_PATHOLOGIC_TUMOR_STAGE' in df_clinical.columns:
            df_clinical['SUBTYPE'] = df_clinical['AJCC_PATHOLOGIC_TUMOR_STAGE'].apply(
                lambda x: 'Early' if 'I' in str(x) or 'II' in str(x) else 'Advanced'
            )
            print(f"  ✓ Created binary subtype based on tumor stage")
        else:
            print("  ! Could not find suitable subtype column")
            print("  ! Available columns:", df_clinical.columns.tolist())
            raise ValueError("No suitable subtype column found")
    
    # Create patient to subtype mapping
    if 'PATIENT_ID' in df_clinical.columns:
        pat2subtype = dict(zip(df_clinical['PATIENT_ID'], df_clinical['SUBTYPE']))
    else:
        pat2subtype = dict(zip(df_clinical.iloc[:, 0], df_clinical['SUBTYPE']))
    
    # Remove patients with missing subtypes
    pat2subtype = {k: v for k, v in pat2subtype.items() if pd.notna(v) and v != 'NA'}
    
    print(f"  ✓ Assigned subtypes to {len(pat2subtype)} patients")
    
    # Show subtype distribution
    subtype_counts = pd.Series(pat2subtype).value_counts()
    print("\n  Subtype distribution:")
    for subtype, count in subtype_counts.items():
        print(f"    {subtype}: {count} patients")
    
except Exception as e:
    print(f"  ✗ Error parsing subtypes: {e}")
    print("  ! Make sure 'data_clinical_patient.txt' is in 'coadread_raw/' folder")
    exit(1)

# ==================================================
# STEP 6: Filter Patients with Both Data and Labels
# ==================================================
print("\nStep 6: Filtering patients...")

# Keep only patients with both alteration data and subtype labels
patients_with_data = set(df_combined.index)
patients_with_labels = set(pat2subtype.keys())
patients_final = patients_with_data & patients_with_labels

print(f"  ✓ Patients with mutation data: {len(patients_with_data)}")
print(f"  ✓ Patients with subtype labels: {len(patients_with_labels)}")
print(f"  ✓ Patients with both: {len(patients_final)}")

# Filter the combined matrix
df_combined = df_combined.loc[sorted(patients_final), :]

# Filter genes: keep genes mutated in at least 4 patients (same as BRCA)
min_mutations = 4
gene_counts = (df_combined > 0).sum(axis=0)
genes_to_keep = gene_counts[gene_counts >= min_mutations].index
df_combined = df_combined.loc[:, genes_to_keep]

print(f"  ✓ After filtering genes (≥{min_mutations} mutations): {df_combined.shape[1]} genes")
print(f"  ✓ Final matrix: {df_combined.shape[0]} patients × {df_combined.shape[1]} genes")

# ==================================================
# STEP 7: Split into Training and Validation Sets
# ==================================================
print("\nStep 7: Splitting into training/validation sets...")

patients = list(df_combined.index)
labels = [pat2subtype[p] for p in patients]

# Stratified split to maintain subtype proportions
train_patients, val_patients, train_labels, val_labels = train_test_split(
    patients, labels, train_size=TRAIN_SPLIT, stratify=labels, random_state=42
)

print(f"  ✓ Training set: {len(train_patients)} patients")
print(f"  ✓ Validation set: {len(val_patients)} patients")

# Show subtype distribution in splits
print("\n  Training set subtypes:")
for subtype, count in pd.Series(train_labels).value_counts().items():
    print(f"    {subtype}: {count}")

print("\n  Validation set subtypes:")
for subtype, count in pd.Series(val_labels).value_counts().items():
    print(f"    {subtype}: {count}")

# ==================================================
# STEP 8: Save Output Files
# ==================================================
print("\nStep 8: Saving output files...")

# Training data: patients × genes matrix
df_train = df_combined.loc[train_patients, :]
df_train.to_csv(OUTPUT_DIR + 'COADREAD_training_data.txt', sep='\t')
print(f"  ✓ Saved: {OUTPUT_DIR}COADREAD_training_data.txt")

# Validation data
df_val = df_combined.loc[val_patients, :]
df_val.to_csv(OUTPUT_DIR + 'COADREAD_validation_data.txt', sep='\t')
print(f"  ✓ Saved: {OUTPUT_DIR}COADREAD_validation_data.txt")

# Training labels
with open(OUTPUT_DIR + 'COADREAD_training_labels.txt', 'w') as f:
    for label in train_labels:
        f.write(f"{label}\n")
print(f"  ✓ Saved: {OUTPUT_DIR}COADREAD_training_labels.txt")

# Validation labels
with open(OUTPUT_DIR + 'COADREAD_validation_labels.txt', 'w') as f:
    for label in val_labels:
        f.write(f"{label}\n")
print(f"  ✓ Saved: {OUTPUT_DIR}COADREAD_validation_labels.txt")

# Feature names (gene names)
with open(OUTPUT_DIR + 'COADREAD_feature_names.txt', 'w') as f:
    for gene in df_combined.columns:
        f.write(f"{gene}\n")
print(f"  ✓ Saved: {OUTPUT_DIR}COADREAD_feature_names.txt")

# ==================================================
# SUMMARY
# ==================================================
print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print(f"\nFinal dataset:")
print(f"  - Total patients: {df_combined.shape[0]}")
print(f"  - Total genes: {df_combined.shape[1]}")
print(f"  - Training samples: {len(train_patients)}")
print(f"  - Validation samples: {len(val_patients)}")
print(f"  - Number of subtypes: {len(set(labels))}")
print(f"\nOutput files saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("  1. Copy BRCA edge features file to use same network:")
print("     cp data/BRCA_edge2features_2.txt coadread_processed/COADREAD_edge2features.txt")
print("  2. Or create COADREAD-specific edge features (advanced)")
print("  3. Run NBS² training with these files!")
print("="*60)
