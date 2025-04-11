# %%
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['HF_DATASETS_CACHE'] = '/SSD/yeongchan/.cache/huggingface/datasets'

# data modules
from data.load_data import load_datasets
from data.normalization import normalize_continuous_variables
from data.tokenize import create_dataset, preprocess_dataset

# model modules
from model.load_model import load_model_dae, load_tokenizers

# trainer modules
from trainer.collator import FTTDataCollatorForDenoisingAutoEncoder
from trainer.train import train_model
import pandas as pd

DAE_MODEL_PATH = "model/20250407_1_UKB_CVD_MLM_DAE/"

# Make directory if not exists
os.makedirs(DAE_MODEL_PATH, exist_ok=True)

# Step 1: Load pretrained model and tokenizers
# In DAE scenario, we often initialize from an MLM pretrained model:
pretrained_path = "model/20250407_1_UKB_CVD_MLM/"
model = load_model_dae(pretrained_model_path=pretrained_path)

# pass DAE_MODEL_PATH so tokenizers get saved here
weights_tokenizer, bias_tokenizer = load_tokenizers(model_path=DAE_MODEL_PATH)

# Step 2: Load datasets and variable names
train, val, test, num_vars, cat_vars = load_datasets()

# Step 3: Normalize continuous variables (and save min-max info)
save_path = os.path.join(DAE_MODEL_PATH, 'train_continuous_variables_min_max_values.json')
train, val, test = normalize_continuous_variables(
    [train, val, test],
    variable_names=num_vars,
    min_new=1,
    max_new=3,
    save_path=save_path
)

# Step 4: Convert DataFrame → Dataset and tokenize
train_dataset = create_dataset(train, num_vars, cat_vars)
val_dataset = create_dataset(val, num_vars, cat_vars)

tokenized_train = preprocess_dataset(train_dataset, weights_tokenizer, bias_tokenizer, config=model.config)
tokenized_val = preprocess_dataset(val_dataset, weights_tokenizer, bias_tokenizer, config=model.config)

# Step 5: Load variable info for noise injection and create collator
var_info_path = "data/ukb_CVD_variable_information_denoising_autoencoder.csv"
var_info = pd.read_csv(var_info_path)

collator = FTTDataCollatorForDenoisingAutoEncoder(
    weights_tokenizer=weights_tokenizer,
    bias_tokenizer=bias_tokenizer,
    var_info=var_info,
    noise_probability=0.2,
    apply_noise_input=False
)

# Step 6: Train the model
trainer = train_model(
    model=model,
    tokenized_train=tokenized_train,
    tokenized_val=tokenized_val,
    data_collator=collator,
    model_path=DAE_MODEL_PATH  # ensure Trainer saves to DAE path
)

# %%
