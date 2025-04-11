#%%
import os
# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_DATASETS_CACHE'] = '/SSD/yeongchan/.cache/huggingface/datasets'

from model.load_model import load_model_mlm, load_tokenizers
from data.load_data import load_datasets
from data.normalization import normalize_continuous_variables
from data.tokenize import create_dataset, preprocess_dataset
from utils.helper_functions import mean_pooling, cls_pooling

import torch
import pandas as pd
from tqdm import tqdm
import json
import numpy as np

# Define paths
MLM_MODEL_PATH = "model/pretrained_model_with_mlm/"
IMPUTED_PATH = os.path.join(MLM_MODEL_PATH, "imputed")
#IMPUTED_PATH = "../model/20240531_1_UKB_CVD_MLM/imputed/"
DEVEL_CSV = "devel_imputed.csv"
TEST_CSV = "test_imputed.csv"

# Step 1: Load pretrained model
model = load_model_mlm(
    model_path=MLM_MODEL_PATH,
    load_init_model=False,
    save_init=False,
    attn_implementation=None,
    )
model.cuda(); model.eval()
config = model.config
print(f"[INFO] Loaded model with {model.num_parameters():,} parameters")

# Step 2: Load imputed data
train = pd.read_csv(os.path.join(IMPUTED_PATH, DEVEL_CSV))
test = pd.read_csv(os.path.join(IMPUTED_PATH, TEST_CSV))

# Extract variable names
num_vars = [col for col in train.columns if col.endswith("_Continuous") or col.endswith("_Integer")]
cat_vars = [col for col in train.columns if col.endswith("_Categorical")]

# Filter only necessary columns
my_vars = ["f.eid", "CVD"] + num_vars + cat_vars
train = train[my_vars]
test = test[my_vars]

# Step 3: Normalize continuous variables using saved min-max values
min_max_path = os.path.join(MLM_MODEL_PATH, "train_continuous_variables_min_max_values.json")
train, test = normalize_continuous_variables(
    [train, test],
    variable_names=num_vars,
    min_new=1,
    max_new=3,
    read_path=min_max_path
)

# Step 4: Create HuggingFace Datasets
mask_token = "[PAD]"  # PAD token (no masking applied here yet)
train_dataset = create_dataset(train, num_vars, cat_vars, special_token_for_missing=mask_token)
test_dataset = create_dataset(test, num_vars, cat_vars, special_token_for_missing=mask_token)

# Step 5: Tokenize the datasets
weights_tokenizer, bias_tokenizer = load_tokenizers(model_path=MLM_MODEL_PATH)

tokenized_train = preprocess_dataset(train_dataset, weights_tokenizer, bias_tokenizer, config=config)
tokenized_test = preprocess_dataset(test_dataset, weights_tokenizer, bias_tokenizer, config=config)

# Step 6: Predict age using masked token approach

#%%
def predict_age(tokenized_dataset, raw_dataset, dataset_name, output_dir, batch_size=512):
    tmp_num_var = "AGE_Continuous"
    print(f"[INFO] Predicting {tmp_num_var} for {dataset_name} set...")
    pred_df = pd.DataFrame()

    for s_idx in tqdm(range(0, len(tokenized_dataset), batch_size), desc=f"Predicting {dataset_name}"):
        e_idx = min(s_idx + batch_size, len(tokenized_dataset))
        tmp_data = tokenized_dataset[s_idx:e_idx]
        target_token = bias_tokenizer.encode(tmp_num_var)[1]
        target_token_indices = [x.index(target_token) for x in tmp_data['num_variable_ids']]

        person_ids = []
        true_y = []
        remove_indices = []

        for i in range(len(tmp_data['num_input_ids'])):
            true_y.append(tmp_data['num_input_val'][i][target_token_indices[i]])
            person_ids.append(raw_dataset[s_idx+i]['f.eid'])
            if tmp_data['num_input_ids'][i][target_token_indices[i]] == weights_tokenizer.pad_token_id:
                remove_indices.append(i)
                true_y.pop()
                person_ids.pop()
            tmp_data['num_input_ids'][i][target_token_indices[i]] = weights_tokenizer.mask_token_id
            tmp_data['num_input_val'][i][target_token_indices[i]] = 1.0

        # remove invalid samples
        for key in tmp_data.keys():
            tmp_data[key] = [x for i, x in enumerate(tmp_data[key]) if i not in remove_indices]

        for key in tmp_data.keys():
            tmp_data[key] = torch.tensor(tmp_data[key]).cuda()

        tmp_data['num_input_val'] = tmp_data['num_input_val'].type(torch.bfloat16)
        tmp_data['num_input_ids'] = tmp_data['num_input_ids'].type(torch.long)
        tmp_data['num_variable_ids'] = tmp_data['num_variable_ids'].type(torch.long)
        tmp_data['cat_input_ids'] = tmp_data['cat_input_ids'].type(torch.long)
        tmp_data['cat_variable_ids'] = tmp_data['cat_variable_ids'].type(torch.long)
        tmp_data['attention_mask'] = tmp_data['attention_mask'].type(torch.long)

        with torch.no_grad():
            outputs = model(**tmp_data)
            logits = outputs['continuous_prediction_logits']
            mask_token_logits = logits[:, target_token_indices[0]]

        tmp_pred_df = pd.DataFrame({
            'f.eid': [str(x) for x in person_ids],
            'TRUE_Y': true_y,
            'PRED_Y': mask_token_logits.tolist(),
        })

        pred_df = pd.concat([pred_df, tmp_pred_df], axis=0)

    save_path = os.path.join(output_dir, f"{dataset_name}_predicted_age.csv")
    pred_df.to_csv(save_path, index=False)
    print(f"[INFO] Saved predictions to {save_path}")
    print(pred_df[['TRUE_Y', 'PRED_Y']].astype(float).corr())

# Run predictions
predict_age(tokenized_train, train_dataset, "devel", MLM_MODEL_PATH)
predict_age(tokenized_test, test_dataset, "test", MLM_MODEL_PATH)


# %%
