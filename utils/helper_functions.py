import torch


def mean_pooling(model_output, attention_mask):
    """
    Perform mean pooling on model output using attention mask.
    """
    token_embeddings = model_output[0]  # shape: (batch_size, seq_len, hidden_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def cls_pooling(model_output):
    """
    Return CLS token embedding from model output.
    """
    return model_output[0][:, 0, :]


def custom_min_max_denormalization(data, X_min, X_max, min_new=1, max_new=3):
    """
    Apply custom min-max denormalization on a list of values.
    """
    return [(x - min_new) / (max_new - min_new) * (X_max - X_min) + X_min for x in data]


def denormalize_continuous_variables(datasets, variable_names, min_max_values, min_new=1, max_new=3):
    """
    Denormalize specified continuous variables in each dataset using stored min/max values.
    """
    for dataset in datasets:
        for var in variable_names:
            X_min, X_max = min_max_values[var]
            dataset[var] = custom_min_max_denormalization(dataset[var], X_min, X_max, min_new, max_new)
    return datasets
