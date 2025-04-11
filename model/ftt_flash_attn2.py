"""
 Convert DistilBERT to use gene embedding.

 Embedding layer 
    add :: 
    rs_ids: Optional[torch.LongTensor] = None, # add rs_ids
    self.snplist_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

BertSelfAttention -> maybe DistilBertFlashAttention2
    add ::
    snp_pos: Optional[torch.LongTensor] = None, # add snp_pos
    genetic distance 계산 고민 필요


"""

from typing import Dict, List, Optional, Set, Tuple, Union

from transformers.models.distilbert.modeling_distilbert import (
    Embeddings,
    DistilBertModel,
    Transformer,
    DistilBertForMaskedLM,
    DistilBertForSequenceClassification
)
from transformers import PretrainedConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput
)
from transformers.utils import logging
from transformers.activations import get_activation

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F



logger = logging.get_logger(__name__)

class TableEmbeddings(Embeddings):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.variable_weights_embeddings = nn.Embedding(config.var_weights_size, config.dim, padding_idx=config.pad_token_id)
        self.variable_bias_embeddings = nn.Embedding(config.max_position_embeddings, config.dim, padding_idx=config.pad_token_id)
        if config.sinusoidal_pos_embds:
            # raise error : we don't use sinusoidal position embeddings in gene embedding
            raise NotImplementedError("We don't use sinusoidal position embeddings in gene embedding")

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(
        self, 
        num_input_val: Optional[torch.Tensor] = None, # add num_input_val
        num_input_ids: Optional[torch.Tensor] = None, # add num_input_ids
        cat_input_ids: Optional[torch.Tensor] = None, # add cat_input_ids
        num_variable_ids: Optional[torch.Tensor] = None, # add num_variable_ids
        cat_variable_ids: Optional[torch.Tensor] = None, # add cat_variable_ids
        input_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters:
            input_ids (torch.Tensor):
                torch.tensor(bs, max_seq_length) The token ids to embed.
            input_embeds (*optional*, torch.Tensor):
                The pre-computed word embeddings. Can only be passed if the input ids are `None`.


        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        
        if num_input_ids is None and cat_input_ids is None:
            #raise error
            raise ValueError("any num_input_ids or cat_input_ids is required")
                
        if num_input_ids is not None:
            num_embed = self.variable_weights_embeddings(num_input_ids)
            num_embed = torch.einsum("bl,bld->bld", num_input_val, num_embed) # num_input_val (bs, num_input_val) 
        if cat_input_ids is not None:
            cat_embed = self.variable_weights_embeddings(cat_input_ids)

        if num_input_val is None and cat_input_ids is not None:
            weights_embeddings = cat_embed
            bias_embeddings = self.variable_bias_embeddings(cat_variable_ids)
        elif num_input_val is not None and cat_input_ids is None:
            weights_embeddings = num_embed
            bias_embeddings = self.variable_bias_embeddings(num_variable_ids)
        else:
            weights_embeddings = torch.cat([num_embed, cat_embed], dim=1)
            bias_embeddings = self.variable_bias_embeddings(torch.cat([num_variable_ids, cat_variable_ids], dim=1))

        embeddings = weights_embeddings + bias_embeddings # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)

        return embeddings


class FTTModel(DistilBertModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.embeddings = TableEmbeddings(config)  # Embeddings
        self.transformer = Transformer(config)  # Encoder
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self._use_sdpa = config._attn_implementation == "sdpa"

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        num_input_val: Optional[torch.Tensor] = None, # add num_input_val
        num_input_ids: Optional[torch.Tensor] = None, # add num_input_ids
        cat_input_ids: Optional[torch.Tensor] = None, # add cat_input_ids
        num_variable_ids: Optional[torch.Tensor] = None, # add num_variable_ids
        cat_variable_ids: Optional[torch.Tensor] = None, # add cat_variable_ids
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutput, Tuple[torch.Tensor, ...]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is not None:
            # raise error
            raise ValueError("inputs_embeds is not supported")
        elif num_input_val is None and cat_input_ids is None:
            # raise error
            raise ValueError("any num_input_ids or cat_input_ids is required")
        
        if num_input_val is not None and cat_input_ids is not None:
            input_shape = torch.tensor(num_input_val.size()) # (bs, num_input_val_length)
            input_shape[1] = input_shape[1] + cat_input_ids.size(1)

        head_mask_is_none = head_mask is None
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embeddings = self.embeddings(
            num_input_val, # add num_input_val
            num_input_ids, # add num_input_ids
            cat_input_ids, # add cat_input_ids
            num_variable_ids, # add num_variable_ids
            cat_variable_ids, # add cat_variable_ids
        )  # (bs, seq_length, dim)

        device = embeddings.device

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)  # (bs, seq_length)

            if self._use_sdpa and head_mask_is_none and not output_attentions:
                attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embeddings.dtype, tgt_len=input_shape[1]
                )

        return self.transformer(
            x=embeddings,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

#not implemented yet.
class IgnoreMaskingMSELoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(IgnoreMaskingMSELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        # 마스킹된 값(예: ignore_index=-100)을 무시하고 손실을 계산합니다.
        mask = target != self.ignore_index
        filtered_pred = prediction[mask]
        filtered_target = target[mask]
        return F.mse_loss(filtered_pred, filtered_target)

class FTTForMaskedLM(DistilBertForMaskedLM):
    _tied_weights_keys = ["vocab_projector.weight"]

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.activation = get_activation(config.activation)

        self.distilbert = FTTModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)
        self.num_projector = nn.Linear(config.dim, 1)

        # Initialize weights and apply final processing
        self.post_init()

        self.mlm_loss_fct = nn.CrossEntropyLoss()
        self.mlm_loss_mse = IgnoreMaskingMSELoss()

    def forward(
        self,
        num_input_val: Optional[torch.Tensor] = None, # add num_input_val
        num_input_ids: Optional[torch.Tensor] = None, # add num_input_ids
        cat_input_ids: Optional[torch.Tensor] = None, # add cat_input_ids
        num_variable_ids: Optional[torch.Tensor] = None, # add num_variable_ids
        cat_variable_ids: Optional[torch.Tensor] = None, # add cat_variable_ids
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.distilbert(
            num_input_val=num_input_val, # add num_input_val
            num_input_ids=num_input_ids, # add num_input_ids
            cat_input_ids=cat_input_ids, # add cat_input_ids
            num_variable_ids=num_variable_ids, # add num_variable_ids
            cat_variable_ids=cat_variable_ids, # add cat_variable_ids
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        
        # NOTE: seq_length has lengths for first continuous variables and then categorical variables
        # So, we computed MSE loss for continuous variables and CrossEntropy loss for categorical variables
        # seq_length in prediction_logits has to divide into continuous and categorical variables according to a index (length of num_variable_ids)
        
        # continuous variables
        continuous_prediction_logits = self.num_projector(prediction_logits[:, :num_variable_ids.size(1), :]) # (bs, 0 to continuous_variable_start idx, 1)
        # squeeze last dim for mse loss
        continuous_prediction_logits = continuous_prediction_logits.squeeze(-1)

        # categorical variables
        categorical_prediction_logits = self.vocab_projector(prediction_logits[:, num_variable_ids.size(1):, :]) # (bs, categorical_variable_start idx to maximum_position_embeddings, vocab_size)

        mlm_loss = None
        if labels is not None:
            continuous_labels = labels[:, :num_variable_ids.size(1)] # (bs, 0 to continuous_variable_start idx)
            continuous_mlm_loss = self.mlm_loss_mse(continuous_prediction_logits, continuous_labels)
            categorical_labels = labels[:, num_variable_ids.size(1):].type(torch.LongTensor).to(device=self.distilbert.device) # (bs, categorical_variable_start idx to maximum_position_embeddings)
            categorical_mlm_loss = self.mlm_loss_fct(categorical_prediction_logits.view(-1, categorical_prediction_logits.size(-1)), categorical_labels.reshape(-1))
            mlm_loss = continuous_mlm_loss + categorical_mlm_loss
        else:
            return {
                'continuous_prediction_logits': continuous_prediction_logits, 
                'categorical_prediction_logits': categorical_prediction_logits
            }
        
        return MaskedLMOutput(
            loss=mlm_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )

class FTTForDenoisingAutoEncoder(DistilBertForMaskedLM):
    _tied_weights_keys = ["vocab_projector.weight"]

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.activation = get_activation(config.activation)

        self.distilbert = FTTModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)
        self.num_projector = nn.Linear(config.dim, 1)

        # Initialize weights and apply final processing
        self.post_init()

        self.dae_loss_fct = nn.CrossEntropyLoss()
        self.dae_loss_mse = IgnoreMaskingMSELoss()

    def forward(
        self,
        num_input_val: Optional[torch.Tensor] = None, # add num_input_val
        num_input_ids: Optional[torch.Tensor] = None, # add num_input_ids
        cat_input_ids: Optional[torch.Tensor] = None, # add cat_input_ids
        num_variable_ids: Optional[torch.Tensor] = None, # add num_variable_ids
        cat_variable_ids: Optional[torch.Tensor] = None, # add cat_variable_ids
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        dlbrt_output = self.distilbert(
            num_input_val=num_input_val, # add num_input_val
            num_input_ids=num_input_ids, # add num_input_ids
            cat_input_ids=cat_input_ids, # add cat_input_ids
            num_variable_ids=num_variable_ids, # add num_variable_ids
            cat_variable_ids=cat_variable_ids, # add cat_variable_ids
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        
        # NOTE: seq_length has lengths for first continuous variables and then categorical variables
        # So, we computed MSE loss for continuous variables and CrossEntropy loss for categorical variables
        # seq_length in prediction_logits has to divide into continuous and categorical variables according to a index (length of num_variable_ids)
        
        # continuous variables
        continuous_prediction_logits = self.num_projector(prediction_logits[:, :num_variable_ids.size(1), :]) # (bs, 0 to continuous_variable_start idx, 1)
        # squeeze last dim for mse loss
        continuous_prediction_logits = continuous_prediction_logits.squeeze(-1)

        # categorical variables
        categorical_prediction_logits = self.vocab_projector(prediction_logits[:, num_variable_ids.size(1):, :]) # (bs, categorical_variable_start idx to maximum_position_embeddings, vocab_size)

        dae_loss = None
        if labels is not None:
            continuous_labels = labels[:, :num_variable_ids.size(1)] # (bs, 0 to continuous_variable_start idx)
            continuous_dae_loss = self.dae_loss_mse(continuous_prediction_logits, continuous_labels)
            categorical_labels = labels[:, num_variable_ids.size(1):].type(torch.LongTensor).to(device=self.distilbert.device) # (bs, categorical_variable_start idx to maximum_position_embeddings)
            categorical_dae_loss = self.dae_loss_fct(categorical_prediction_logits.view(-1, categorical_prediction_logits.size(-1)), categorical_labels.reshape(-1))
            dae_loss = continuous_dae_loss + categorical_dae_loss
        else:
            return {
                'continuous_prediction_logits': continuous_prediction_logits, 
                'categorical_prediction_logits': categorical_prediction_logits
            }
        
        return MaskedLMOutput(
            loss=dae_loss,
            logits=prediction_logits,
            hidden_states=dlbrt_output.hidden_states,
            attentions=dlbrt_output.attentions,
        )


class FTTForSequenceClassification(DistilBertForSequenceClassification):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = FTTModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        num_input_val: Optional[torch.Tensor] = None, # add num_input_val
        num_input_ids: Optional[torch.Tensor] = None, # add num_input_ids
        cat_input_ids: Optional[torch.Tensor] = None, # add cat_input_ids
        num_variable_ids: Optional[torch.Tensor] = None, # add num_variable_ids
        cat_variable_ids: Optional[torch.Tensor] = None, # add cat_variable_ids
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            num_input_val=num_input_val, # add num_input_val
            num_input_ids=num_input_ids, # add num_input_ids
            cat_input_ids=cat_input_ids, # add cat_input_ids
            num_variable_ids=num_variable_ids, # add num_variable_ids
            cat_variable_ids=cat_variable_ids, # add cat_variable_ids
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
