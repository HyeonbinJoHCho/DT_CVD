import numpy as np
import pandas as pd
from datasets import Dataset

def create_dataset(df, numerical_variable_names, categorical_variable_names, special_token_for_missing="[PAD]"):
    """
    Convert a DataFrame into HuggingFace Dataset using a generator.

    Args:
        df (pd.DataFrame): Input DataFrame
        numerical_variable_names (List[str]): List of continuous/integer variables
        categorical_variable_names (List[str]): List of categorical variables
        special_token_for_missing (str): Token to use for missing variables

    Returns:
        HuggingFace Dataset object
    """
    def generic_gen(df):
        for i in range(len(df)):
            tmp = df.loc[i]
            y = tmp["CVD"]
            x = tmp[numerical_variable_names + categorical_variable_names]
            x = pd.DataFrame(x).T
            if x.dropna(axis=1).empty:
                continue

            x_num_vars = numerical_variable_names
            x_num_input = x_num_vars
            x_num = x[x_num_vars].values.tolist()[0]
            missing_idx_num = [i for i, v in enumerate(x_num) if pd.isna(v)]
            if missing_idx_num:
                x_num_input = [x_num_vars[i] if i not in missing_idx_num else special_token_for_missing for i in range(len(x_num_vars))]
                x_num = [x_num[i] if i not in missing_idx_num else 1 for i in range(len(x_num))]

            x_num_vars = ' '.join(x_num_vars)
            x_num_input = ' '.join(x_num_input)

            x_cat_vars = categorical_variable_names
            x_cat = x[x_cat_vars].values.tolist()[0]
            missing_idx_cat = [i for i, v in enumerate(x_cat) if pd.isna(v)]
            x_cat = [x_cat_vars[i] + "_" + str(int(x_cat[i])) if i not in missing_idx_cat else special_token_for_missing for i in range(len(x_cat))]
            x_cat = ' '.join(x_cat)
            x_cat_vars = ' '.join(x_cat_vars)

            yield {
                'x_num_input': x_num_input,
                'x_num_vars': x_num_vars,
                'x_num_val': x_num,
                'x_cat_vars': x_cat_vars,
                'x_cat': x_cat,
                'y': y,
                'f.eid': tmp["f.eid"]
            }

    return Dataset.from_generator(generator=generic_gen, gen_kwargs={"df": df}, num_proc=16).shuffle(seed=1234)

def preprocess_dataset(dataset, weights_tokenizer, bias_tokenizer, config=None):
    if config is None:
        config = weights_tokenizer.init_kwargs['config']
    def preprocess_function(examples):
        return_dict = {}
        return_dict['num_input_ids'] = weights_tokenizer([x for x in examples["x_num_input"]], add_special_tokens=False)['input_ids']
        return_dict['num_input_ids'] = [[1] + sublist for sublist in return_dict['num_input_ids']]
        return_dict['num_variable_ids'] = bias_tokenizer([x for x in examples["x_num_vars"]], add_special_tokens=False)['input_ids']
        return_dict['num_variable_ids'] = [[1] + sublist for sublist in return_dict['num_variable_ids']]
        return_dict['num_input_val'] = [[1] + x for x in examples["x_num_val"]]

        return_dict['cat_input_ids'] = weights_tokenizer([x for x in examples["x_cat"]],
                                                          padding='max_length',
                                                          truncation=True,
                                                          max_length=config.max_position_embeddings)['input_ids']
        return_dict['cat_input_ids'] = [cat_sublist[1:(config.max_position_embeddings - len(num_sublist) + 1)]
                                        for cat_sublist, num_sublist in zip(return_dict['cat_input_ids'], return_dict['num_input_ids'])]

        return_dict['cat_variable_ids'] = bias_tokenizer([x for x in examples["x_cat_vars"]],
                                                          padding='max_length',
                                                          truncation=True,
                                                          max_length=config.max_position_embeddings)['input_ids']
        return_dict['cat_variable_ids'] = [cat_sublist[1:(config.max_position_embeddings - len(num_sublist) + 1)]
                                           for cat_sublist, num_sublist in zip(return_dict['cat_variable_ids'], return_dict['num_variable_ids'])]

        return_dict['attention_mask'] = [[1] * (len(num_string.split(' ')) + len(cat_string.split(' ')) + 2) +
                                         [0] * (config.max_position_embeddings - len(num_string.split(' ')) - len(cat_string.split(' ')) - 2)
                                         for num_string, cat_string in zip(examples["x_num_vars"], examples["x_cat_vars"])]
        return return_dict

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=16,
        remove_columns=dataset.column_names
    )
    return tokenized_dataset


