# %%
import os
# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_DATASETS_CACHE'] = '/SSD/yeongchan/.cache/huggingface/datasets'

from model.ftt_flash_attn2 import FTTModel
from data.normalization import normalize_continuous_variables
from data.tokenize import create_dataset
from model.load_model import load_tokenizers
from data.tokenize import preprocess_dataset
from utils.helper_functions import mean_pooling, cls_pooling

import torch
import pandas as pd
import os
from tqdm import tqdm


DAE_MODEL_PATH = "model/pretrained_model_with_dae/"

# Define paths
IMPUTED_PATH = os.path.join("model/pretrained_model_with_mlm/imputed/")
DEVEL_CSV = "devel_imputed.csv"
TEST_CSV = "test_imputed.csv"

# Step 1: Load pretrained FTTModel
model = FTTModel.from_pretrained(DAE_MODEL_PATH)
config = model.config
# Log number of parameters
print(f"[INFO] Loaded FTTModel with {model.num_parameters():,} parameters")
model.cuda()
model.eval()

# Step 2: Load imputed data
train = pd.read_csv(os.path.join(IMPUTED_PATH, DEVEL_CSV))
test = pd.read_csv(os.path.join(IMPUTED_PATH, TEST_CSV))

# Extract variable names
num_vars = [col for col in train.columns if col.endswith("_Continuous") or col.endswith("_Integer")]
cat_vars = [col for col in train.columns if col.endswith("_Categorical")]

# Select only relevant columns
my_vars = ["f.eid", "CVD"] + num_vars + cat_vars
train = train[my_vars]
test = test[my_vars]

# Step 3: Use pre-saved min-max normalization values from MLM training phase
MIN_MAX_JSON_PATH = os.path.join(DAE_MODEL_PATH, "train_continuous_variables_min_max_values.json")

train, test = normalize_continuous_variables(
    [train, test],
    variable_names=num_vars,
    min_new=1,
    max_new=3,
    read_path=MIN_MAX_JSON_PATH
)

# Step 4: Create HuggingFace Datasets from DataFrames
mask_token = "[PAD]"  # PAD is used here because no masking is needed for inference
train_dataset = create_dataset(train, num_vars, cat_vars, special_token_for_missing=mask_token)
test_dataset = create_dataset(test, num_vars, cat_vars, special_token_for_missing=mask_token)

print(f"[INFO] Created HuggingFace datasets: {len(train_dataset)} train, {len(test_dataset)} test")

# Load tokenizers from DAE path (shared with MLM)
weights_tokenizer, bias_tokenizer = load_tokenizers(model_path=DAE_MODEL_PATH)

# Step 5: Tokenize HuggingFace Datasets
tokenized_train = preprocess_dataset(train_dataset, weights_tokenizer, bias_tokenizer, config=config)
tokenized_test = preprocess_dataset(test_dataset, weights_tokenizer, bias_tokenizer, config=config)

print(f"[INFO] Tokenized datasets: train={len(tokenized_train)}, test={len(tokenized_test)}")

def extract_embeddings(tokenized_dataset, raw_dataset, output_path, batch_size=1024):
    all_df = pd.DataFrame()
    for i in tqdm(range(0, len(tokenized_dataset), batch_size), desc=f"Embedding: {output_path}"):
        end = min(i + batch_size, len(tokenized_dataset))
        batch = {k: torch.tensor(tokenized_dataset[k][i:end]).cuda() for k in tokenized_dataset.column_names}
        batch['attention_mask'] = batch['attention_mask'].type(torch.float32)
        f_eid = [str(int(eid)) for eid in raw_dataset[i:end]['f.eid']]

        with torch.no_grad():
            outputs = model(**batch)

        mean_embed = mean_pooling(outputs, batch["attention_mask"]).cpu().numpy()
        cls_embed = cls_pooling(outputs).cpu().numpy()

        df_mean = pd.DataFrame(mean_embed, columns=[f"MEAN{i+1}" for i in range(mean_embed.shape[1])])
        df_cls = pd.DataFrame(cls_embed, columns=[f"CLS{i+1}" for i in range(cls_embed.shape[1])])
        df_eid = pd.DataFrame({'f.eid': f_eid})

        combined = pd.concat([df_eid, df_mean, df_cls], axis=1)
        all_df = pd.concat([all_df, combined], axis=0)

    all_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved embeddings to {output_path}")

# Run embedding extraction
extract_embeddings(tokenized_train, train_dataset, os.path.join(DAE_MODEL_PATH, "devel_imputed_embeddings.csv"))
extract_embeddings(tokenized_test, test_dataset, os.path.join(DAE_MODEL_PATH, "test_imputed_embeddings.csv"))

# %%
