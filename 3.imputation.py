# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_DATASETS_CACHE'] = '/SSD/yeongchan/.cache/huggingface/datasets'


from model.load_model import load_model_mlm, load_tokenizers
from data.load_data import load_datasets
from data.normalization import normalize_continuous_variables
from data.tokenize import create_dataset, preprocess_dataset
from utils.helper_functions import custom_min_max_denormalization, denormalize_continuous_variables

import torch
import pandas as pd
import json
from tqdm import tqdm

# Step 1: Load model and tokenizers
# Define path
MLM_MODEL_PATH = "model/pretrained_model_with_mlm/"

model = load_model_mlm(model_path=MLM_MODEL_PATH,
                       load_init_model=False,
                       save_init=False,
                       attn_implementation="flash_attention_2",)
weights_tokenizer, bias_tokenizer = load_tokenizers(model_path=MLM_MODEL_PATH)

model.cuda()
model.eval()

# Step 2: Load raw data (devel/test split) and extract variable names
train, val, test, num_vars, cat_vars = load_datasets(shuffle=False, train_frac=0.8, seed=1234)

# Step 3: Normalize continuous variables using pre-saved min-max JSON
min_max_path = os.path.join(MLM_MODEL_PATH, "train_continuous_variables_min_max_values.json")
train, val, test = normalize_continuous_variables(
    [train, val, test],
    variable_names=num_vars,
    min_new=1,
    max_new=3,
    read_path=min_max_path
)

# Step 4: Create masked datasets for imputation
mask_token = "[MASK]"
train_dataset = create_dataset(train, num_vars, cat_vars, special_token_for_missing=mask_token)
val_dataset = create_dataset(val, num_vars, cat_vars, special_token_for_missing=mask_token)
test_dataset = create_dataset(test, num_vars, cat_vars, special_token_for_missing=mask_token)

# Step 5: Tokenize the datasets using pretrained tokenizers
tokenized_train = preprocess_dataset(train_dataset, weights_tokenizer, bias_tokenizer, config=model.config)
tokenized_val = preprocess_dataset(val_dataset, weights_tokenizer, bias_tokenizer, config=model.config)
tokenized_test = preprocess_dataset(test_dataset, weights_tokenizer, bias_tokenizer, config=model.config)

# Step 6: Predict masked values and save imputed datasets
# Prepare all data splits to process
all_data = {
    'dataset': [train_dataset, val_dataset, test_dataset],
    'tokenized_dataset': [tokenized_train, tokenized_val, tokenized_test],
    'name': ['train', 'val', 'test']
}

os.makedirs(os.path.join(MLM_MODEL_PATH, "imputed"), exist_ok=True)

for tmp_dataset, tmp_tokenized_dataset, tmp_data_name in zip(all_data['dataset'], all_data['tokenized_dataset'], all_data['name']):
    batch_size = 512
    #divide incides by batch
    start_idx = [i for i in range(0, len(tmp_tokenized_dataset), batch_size)]

    pred_df = pd.DataFrame()
    for s_idx in start_idx:
        if s_idx % 10240 == 0:
            print(str(s_idx) + " / " + str(len(tmp_tokenized_dataset)) + " start!!")
        e_idx = min(s_idx + batch_size, len(tmp_tokenized_dataset))
        tmp_input = tmp_tokenized_dataset[s_idx:e_idx]
        tmptmp_data = tmp_dataset[s_idx:e_idx]

        person_ids = []
        remove_person_ids = []
        for i in range(len(tmp_input['cat_input_ids'])):
            person_ids.append(tmptmp_data['f.eid'][i])
            # if there are no missing values, skip
            if (weights_tokenizer.mask_token_id not in tmp_input['cat_input_ids'][i]) & (weights_tokenizer.mask_token_id not in tmp_input['num_input_ids'][i]):
                remove_person_ids.append(tmptmp_data['f.eid'][i])
        
        # if person_ids is in remove indices, then remove it from person_ids
        person_ids = [x for i, x in enumerate(person_ids) if x not in remove_person_ids]

        # if remove_indices is not empty, remove it from tmp_input       
        if len(remove_person_ids) != 0:
            for key in tmp_input.keys():
                tmp_input[key] = [x for x, i in zip(tmp_input[key], tmptmp_data['f.eid']) if i not in remove_person_ids]

        # if tmp_input is empty, then skip 
        if len(tmp_input['cat_input_ids']) == 0: 
            continue
        
        for key in tmp_input.keys():
            tmp_input[key] = torch.tensor(tmp_input[key]).cuda()

        tmp_input['num_input_val'] = tmp_input['num_input_val'].type(torch.float32)
        tmp_input['num_input_ids'] = tmp_input['num_input_ids'].type(torch.long)
        tmp_input['num_variable_ids'] = tmp_input['num_variable_ids'].type(torch.long)
        tmp_input['cat_input_ids'] = tmp_input['cat_input_ids'].type(torch.long)
        tmp_input['cat_variable_ids'] = tmp_input['cat_variable_ids'].type(torch.long)
        tmp_input['attention_mask'] = tmp_input['attention_mask'].type(torch.long)

        # get mask token indices in num_input_ids and cat_input_ids
        mask_indices_num = torch.where(tmp_input['num_input_ids'] == weights_tokenizer.mask_token_id)
        mask_indices_cat = torch.where(tmp_input['cat_input_ids'] == weights_tokenizer.mask_token_id)
        
        # evaluation
        with torch.no_grad(), torch.amp.autocast('cuda'):
            outputs = model(**tmp_input)
            model_output_con = outputs['continuous_prediction_logits']
            model_output_cat = outputs['categorical_prediction_logits']
            top_1_tokens = torch.topk(model_output_cat, 1, dim=-1).indices.squeeze()
        
        if top_1_tokens.dim() == 1:
            top_1_tokens = top_1_tokens.unsqueeze(0)

        for i in range(mask_indices_num[0].shape[0]):
            tmp_input['num_input_val'][mask_indices_num[0][i], mask_indices_num[1][i]] = model_output_con[mask_indices_num[0][i], mask_indices_num[1][i]]

        for i in range(mask_indices_cat[0].shape[0]):
            tmp_input['cat_input_ids'][mask_indices_cat[0][i], mask_indices_cat[1][i]] = top_1_tokens[mask_indices_cat[0][i], mask_indices_cat[1][i]]
        
        tmp_input_num_val = tmp_input['num_input_val'].cpu().numpy()
        tmp_input_cat_decoded = weights_tokenizer.batch_decode(tmp_input['cat_input_ids'])
        tmp_input_cat_decoded = [x.split(' ') for x in tmp_input_cat_decoded]
        tmp_input_cat_decoded = [x[:len(cat_vars)] for x in tmp_input_cat_decoded]
        tmp_input_num_var_decoded = bias_tokenizer.decode(tmp_input['num_variable_ids'][0]).split(' ')
        tmp_input_cat_var_decoded = bias_tokenizer.decode(tmp_input['cat_variable_ids'][0]).split(' ')
        
        #tmp_input_num_var_decoded and tmp_input_cat_var_decoded will be columns in a tmp_df
        tmp_num_df = pd.DataFrame(tmp_input_num_val[:, 1:], columns=tmp_input_num_var_decoded[1:])
        tmp_cat_df = pd.DataFrame(tmp_input_cat_decoded, columns=tmp_input_cat_var_decoded[:len(cat_vars)])
        tmp_df = pd.DataFrame([int(x) for x in person_ids], columns=['f.eid'])
        tmp_df = pd.concat([tmp_df, tmp_num_df, tmp_cat_df], axis=1)

        pred_df = pd.concat([pred_df, tmp_df], axis=0)

    pred_df.to_csv(os.path.join(MLM_MODEL_PATH, "imputed", f"{tmp_data_name}_pre_imputed.csv"), index=False)


# Step 7: Denormalize and clean the imputed results
# Load pre-imputed data
train_pre = pd.read_csv(os.path.join(MLM_MODEL_PATH, "imputed", "train_pre_imputed.csv"))
val_pre = pd.read_csv(os.path.join(MLM_MODEL_PATH, "imputed", "val_pre_imputed.csv"))
test_pre = pd.read_csv(os.path.join(MLM_MODEL_PATH, "imputed", "test_pre_imputed.csv"))

# Find incomplete person IDs
incomplete_ids = pd.concat([train_pre, val_pre, test_pre])['f.eid'].unique()

# Denormalize continuous variables
with open(min_max_path, 'r') as f:
    min_max_values = json.load(f)

train_pre, val_pre, test_pre = denormalize_continuous_variables(
    [train_pre, val_pre, test_pre],
    num_vars,
    min_max_values=min_max_values,
    min_new=1,
    max_new=3
)

# Remove PAD in categorical
for var in cat_vars:
    for df in [train_pre, val_pre, test_pre]:
        df.drop(df[~df[var].str.contains(f"{var}_", na=False)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

# Convert categorical values back to original labels
for df in [train_pre, val_pre, test_pre]:
    for var in cat_vars:
        df[var] = df[var].str.replace(f"{var}_", "").astype(train[var].dtype)

# Merge pre-imputed with original complete data
devel_pre = pd.concat([train_pre, val_pre], axis=0).reset_index(drop=True)
devel_base = pd.concat([train, val], axis=0).reset_index(drop=True)
devel_complete = devel_base[~devel_base['f.eid'].isin(incomplete_ids)]

devel_imputed = pd.merge(
    devel_base[['f.eid', 'CVD']], devel_pre, on='f.eid', how='inner'
)
devel_imputed = pd.concat([devel_imputed, devel_complete], axis=0).reset_index(drop=True)

test_complete = test[~test['f.eid'].isin(incomplete_ids)]
test_imputed = pd.merge(test[['f.eid', 'CVD']], test_pre, on='f.eid', how='inner')
test_imputed = pd.concat([test_imputed, test_complete], axis=0).reset_index(drop=True)

# Save final outputs
devel_imputed.to_csv(os.path.join(MLM_MODEL_PATH, "imputed", "devel_imputed.csv"), index=False)
test_imputed.to_csv(os.path.join(MLM_MODEL_PATH, "imputed", "test_imputed.csv"), index=False)

# %%
# check if there are any missing values in devel_imputed and test_imputed
devel_imputed.isnull().sum()
test_imputed.isnull().sum()
# %%