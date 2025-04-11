import os
import random
import numpy as np
import pandas as pd
import torch
from transformers import PreTrainedTokenizer
from transformers.data.data_collator import torch_default_data_collator, DefaultDataCollator

class FTTDataCollatorForCopyMaskedLM(DefaultDataCollator):
    """
    Custom DataCollator for Copy-Masked Language Modeling.

    - Applies masking to selected variable tokens (both categorical and numerical).
    - 15% of the samples are randomly selected for variable-wise masking.
    - Standard MLM masking is applied in addition to custom masking.
    """

    def __init__(
        self,
        weights_tokenizer: PreTrainedTokenizer,
        bias_tokenizer: PreTrainedTokenizer,
        masking_pattern: list,
        copy_mask_probability: float = 0.15
    ):
        self.weights_tokenizer = weights_tokenizer
        self.bias_tokenizer = bias_tokenizer
        self.masking_pattern = masking_pattern
        self.copy_mask_probability = copy_mask_probability

    def __call__(self, examples):
        # Collate the batch (convert to tensors and pad as needed)
        batch = torch_default_data_collator(examples)

        # Concatenate numeric and categorical input IDs and variable IDs
        input_ids = torch.cat((batch['num_input_ids'], batch['cat_input_ids']), dim=1)
        variable_ids = torch.cat((batch['num_variable_ids'], batch['cat_variable_ids']), dim=1)
        labels = torch.cat((batch['num_input_val'], batch['cat_input_ids']), dim=1)

        # Find CLS and SEP token positions (for slicing variable regions)
        sep_idx = (variable_ids[0] == self.bias_tokenizer.sep_token_id).nonzero().squeeze(-1).item()
        var_len = sep_idx - 1  # Exclude CLS

        # Extract patient-wise input (before SEP)
        patient_input_ids = input_ids[:, 1:sep_idx]

        # Find padded tokens (value == PAD)
        padded_mask = (patient_input_ids == self.weights_tokenizer.pad_token_id).long()

        # Select masking pattern (random if multiple are given)
        if len(self.masking_pattern) == 1:
            masking_vars = self.masking_pattern[0]
        else:
            masking_vars = random.choice(self.masking_pattern)

        # Encode masking variable names to token IDs
        masking_token_ids = [self.bias_tokenizer.encode(v)[1] for v in masking_vars]

        # Create variable-level mask map (1 if this var is in masking list)
        mask_map = torch.tensor([
            1 if vid in masking_token_ids else 0
            for vid in variable_ids[0, 1:sep_idx]
        ]).unsqueeze(-1)  # Shape: (var_len, 1)

        # Check which patients can be masked (no pre-existing missing in target variables)
        patient_maskable = (padded_mask @ mask_map == 0).squeeze(-1).tolist()
        patient_maskable = [i for i, ok in enumerate(patient_maskable) if ok == 1]

        # Select 15% of maskable patients
        num_mask = int(len(patient_maskable) * self.copy_mask_probability)
        if num_mask > 0:
            masked_patients = random.sample(patient_maskable, num_mask)
        else:
            masked_patients = []

        # Apply copy-masking for selected patients
        for i in masked_patients:
            indices = [j for j in range(var_len) if variable_ids[i, j+1].item() in masking_token_ids]
            for idx in indices:
                input_ids[i, idx + 1] = self.weights_tokenizer.mask_token_id

        # Standard MLM masking on top
        prob_matrix = torch.full(variable_ids.shape, self.copy_mask_probability)
        special_mask = [
            self.bias_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in variable_ids.tolist()
        ]
        prob_matrix.masked_fill_(torch.tensor(special_mask, dtype=torch.bool), value=0.0)
        mlm_mask = torch.bernoulli(prob_matrix).bool()
        input_ids[mlm_mask] = self.weights_tokenizer.mask_token_id

        # Set labels (only masked positions should be predicted)
        masked = input_ids == self.weights_tokenizer.mask_token_id
        labels[~masked] = -100  # Ignore non-masked tokens

        # Split tensors back into original fields
        num_len = batch['num_input_ids'].shape[1]
        batch['num_input_ids'] = input_ids[:, :num_len].long()
        batch['cat_input_ids'] = input_ids[:, num_len:].long()
        batch['num_variable_ids'] = variable_ids[:, :num_len].long()
        batch['cat_variable_ids'] = variable_ids[:, num_len:].long()

        # For masked numeric tokens, set their weight to 1
        batch['num_input_val'][batch['num_input_ids'] == self.weights_tokenizer.mask_token_id] = 1

        # Set final labels (float16 for compatibility with bf16 training)
        batch['labels'] = labels.to(torch.bfloat16)

        return batch


class FTTDataCollatorForDenoisingAutoEncoder(DefaultDataCollator):
    """
    Custom DataCollator for Denoising AutoEncoder.

    - Injects noise into numeric/categorical variables.
    - If apply_noise_input=True, random noise is applied to input.
    - If apply_noise_input=False, original input remains, but only noisy positions are predicted.
    """

    def __init__(self, weights_tokenizer, bias_tokenizer, var_info, noise_probability=0.3, apply_noise_input=True):
        super().__init__(return_tensors="pt")
        self.weights_tokenizer = weights_tokenizer
        self.bias_tokenizer = bias_tokenizer
        self.var_info = var_info
        self.noise_probability = noise_probability
        self.apply_noise_input = apply_noise_input

    def __call__(self, examples):
        # 1. Collate the batch
        batch = torch_default_data_collator(examples)

        # 2. Concatenate numeric & categorical IDs
        input_ids = torch.cat((batch['num_input_ids'], batch['cat_input_ids']), dim=1)
        batch_size = input_ids.shape[0]
        variable_ids = torch.cat((batch['num_variable_ids'], batch['cat_variable_ids']), dim=1)

        labels = torch.cat((batch['num_input_val'], batch['cat_input_ids']), dim=1)
        noise_input = torch.cat((batch['num_input_val'], batch['cat_input_ids']), dim=1)

        # 3. Find the region for variables (CLS to SEP)
        sep_token_idx = torch.tensor(variable_ids[0].tolist().index(self.bias_tokenizer.sep_token_id))
        variable_list = variable_ids[0, 1:sep_token_idx]
        # decode with bias_tokenizer for consistent variable names
        variable_list_decode = self.bias_tokenizer.decode(variable_list).split(' ')

        # 4. Probability matrix for noise injection
        probability_matrix = torch.full(variable_ids.shape, self.noise_probability)
        special_tokens_mask = [
            self.bias_tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in variable_ids.tolist()
        ]
        special_token_indices = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_token_indices, value=0.0)
        noise_indices = torch.bernoulli(probability_matrix).bool()

        # 5. Create random data for noise
        matrix_data = []
        for row in self.var_info.itertuples():
            # row: (Index, var_name, min_val, max_val)
            var_name = row[1]
            min_val, max_val = row[2], row[3]

            if "Continuous" in var_name:
                data = np.random.uniform(low=min_val, high=max_val, size=batch_size)
            elif "Categorical" in var_name:
                data = np.random.randint(low=min_val, high=max_val + 1, size=batch_size)
                # convert to tokens
                data = [self.weights_tokenizer.encode(f"{var_name}_{x}")[1] for x in data]
            else:
                data = np.zeros(batch_size)  # fallback

            matrix_data.append(data)

        random_data_matrix = pd.DataFrame(np.array(matrix_data).T, columns=[row[1] for row in self.var_info.itertuples()])
        random_data_matrix = random_data_matrix[variable_list_decode]
        random_data_matrix = torch.tensor(random_data_matrix.to_numpy(), dtype=torch.float32)

        # put random data into a larger tensor aligned with input_ids shape
        random_data_matrix2 = torch.zeros(size=input_ids.size())
        random_data_matrix2[:, 1:(random_data_matrix.shape[1]+1)] = random_data_matrix

        # 6. apply noise
        noise_input[noise_indices] = random_data_matrix2[noise_indices]

        # 7. split back into numeric/cat
        num_len = batch['num_input_ids'].shape[1]
        num_input_ids = input_ids[:, :num_len]
        cat_input_ids = input_ids[:, num_len:]
        num_variable_ids = variable_ids[:, :batch['num_variable_ids'].shape[1]]
        cat_variable_ids = variable_ids[:, batch['num_variable_ids'].shape[1]:]

        if self.apply_noise_input:
            num_input_val = noise_input[:, :num_len]
            cat_input_ids = noise_input[:, num_len:]
            labels[~noise_indices] = -100
        else:
            num_input_val = batch['num_input_val']
            labels[special_token_indices] = -100

        # 8. update batch
        batch['num_input_ids'] = num_input_ids.long()
        batch['cat_input_ids'] = cat_input_ids.long()
        batch['num_variable_ids'] = num_variable_ids.long()
        batch['cat_variable_ids'] = cat_variable_ids.long()
        batch['num_input_val'] = num_input_val.to(torch.bfloat16)
        batch['labels'] = labels.to(torch.bfloat16)

        return batch

