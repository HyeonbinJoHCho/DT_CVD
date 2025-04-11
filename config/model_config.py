import os
from transformers import PretrainedConfig
from config.paths import CONFIG_PATH

def get_config(task: str = "mlm"):
    config = PretrainedConfig.from_json_file(os.path.join(CONFIG_PATH, "ftt_flash_attn2_config.json"))
    config.max_position_embeddings = 256
    config.vocab_size = config.var_weights_size
    
    if task == "mlm":
        config._attn_implementation = None
    elif task == "dae":
        config._attn_implementation = "flash_attention_2"
    
    return config