#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_DATASETS_CACHE'] = '/SSD/yeongchan/.cache/huggingface/datasets'

from data.load_data import load_datasets
from data.normalization import normalize_continuous_variables
from data.tokenize import create_dataset, preprocess_dataset
from data.missing_pattern import load_missing_pattern

from model.load_model import load_model_mlm, load_tokenizers
from trainer.collator import FTTDataCollatorForCopyMaskedLM
from trainer.train import train_model

import pandas as pd

MLM_MODEL_PATH = "model/20250407_1_UKB_CVD_MLM/"

# Step 1: Load datasets and variable names
train, val, test, num_vars, cat_vars = load_datasets()

# Step 2: Normalize continuous variables (and save min-max info)
save_path = os.path.join(MLM_MODEL_PATH, 'train_continuous_variables_min_max_values.json')
train, val, test = normalize_continuous_variables(
    [train, val, test],
    variable_names=num_vars,
    min_new=1,
    max_new=3,
    save_path=save_path
)

# Step 3: Convert DataFrame → Dataset and tokenize
train_dataset = create_dataset(train, num_vars, cat_vars)
val_dataset = create_dataset(val, num_vars, cat_vars)

# Step 4: Load model and tokenizers
model = load_model_mlm(model_path=MLM_MODEL_PATH, 
                       load_init_model=True, 
                       save_init=True, 
                       attn_implementation="flash_attention_2")
weights_tokenizer, bias_tokenizer = load_tokenizers(model_path=MLM_MODEL_PATH)

tokenized_train = preprocess_dataset(train_dataset, weights_tokenizer, bias_tokenizer, config=model.config)
tokenized_val = preprocess_dataset(val_dataset, weights_tokenizer, bias_tokenizer, config=model.config)

# Step 5: Load masking pattern and create collator
missing_pattern = load_missing_pattern("data/ukb_CVD_missing_variable_information.csv")
collator = FTTDataCollatorForCopyMaskedLM(
    weights_tokenizer=weights_tokenizer,
    bias_tokenizer=bias_tokenizer,
    masking_pattern=missing_pattern,
    copy_mask_probability=0.15
)

# Step 6: Train the model
trainer = train_model(
    model=model,
    tokenized_train=tokenized_train,
    tokenized_val=tokenized_val,
    data_collator=collator,
    model_path=MLM_MODEL_PATH
)


# %%
