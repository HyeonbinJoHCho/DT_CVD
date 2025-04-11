from model.ftt_flash_attn2 import FTTForMaskedLM, FTTForDenoisingAutoEncoder
from config.model_config import get_config

import os
import torch
from transformers import AutoTokenizer

def load_model_mlm(model_path: str, 
                   load_init_model: bool = True, 
                   save_init: bool = True, 
                   attn_implementation=None):
    """
    Load or initialize FTTForMaskedLM model with configuration for MLM task.

    Args:
        model_path (str): Path to save/load the model.
        save_init (bool): Whether to save the initialized model to disk.

    Returns:
        model (FTTForMaskedLM): Loaded model instance
    """

    
    config = get_config(task="mlm")
    if attn_implementation is not None:
        config._attn_implementation = attn_implementation
        
    if load_init_model:
        model_path = os.path.join(model_path, "init_model/")

    if save_init:
        model = FTTForMaskedLM(config=config)
        model.save_pretrained(model_path)

    if attn_implementation is not None:
        model = FTTForMaskedLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
    else:
        model = FTTForMaskedLM.from_pretrained(
            model_path,
            config=config,
        )

    print(f"Number of parameters: {model.num_parameters()}")
    return model

def load_model_dae(pretrained_model_path: str):
    """
    Load pretrained FTTForDenoisingAutoEncoder model with configuration for DAE task.

    Args:
        pretrained_model_path (str): Path to the pretrained model directory.

    Returns:
        model (FTTForDenoisingAutoEncoder): Loaded model instance
    """
    config = get_config(task="dae")
    model = FTTForDenoisingAutoEncoder.from_pretrained(
        pretrained_model_path,
        config=config,
        torch_dtype=torch.bfloat16
    )
    print(f"Number of parameters: {model.num_parameters()}")
    return model

def load_tokenizers(model_path: str):
    """
    Load the variable weights and bias tokenizers, and save copies in model path.

    Args:
        model_path (str): Path to save the tokenizers.

    Returns:
        ftt_var_weights_tokenizer, ftt_var_bias_tokenizer
    """
    weights_tokenizer_path = "model/ftt_variable_weights_tokenizer"
    bias_tokenizer_path = "model/ftt_variable_bias_tokenizer"

    ftt_var_weights_tokenizer = AutoTokenizer.from_pretrained(weights_tokenizer_path)
    ftt_var_bias_tokenizer = AutoTokenizer.from_pretrained(bias_tokenizer_path)

    # Save local copies
    ftt_var_weights_tokenizer.save_pretrained(os.path.join(model_path, "ftt_variable_weights_tokenizer"))
    ftt_var_bias_tokenizer.save_pretrained(os.path.join(model_path, "ftt_variable_bias_tokenizer"))

    return ftt_var_weights_tokenizer, ftt_var_bias_tokenizer

