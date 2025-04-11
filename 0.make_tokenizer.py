#%%
from tokenizers import Tokenizer, models, processors, normalizers, decoders
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace

import pandas as pd

import os

from transformers import PreTrainedTokenizerFast
from transformers import AutoTokenizer

var_info = pd.read_csv("data/ukb_CVD_variable_information.csv")

# defines variables
# y :: "YEAR_5_CVD"
# x :: 
# - continous variables are named with "_Continuous" or "_Integer" suffix
# - categorical variables are named with "_Categorical" suffix
numerical_variable_names = [i for i in var_info['VAR'] if i.endswith("_Continuous") or i.endswith("_Integer")]
categorical_variable_names = [i for i in var_info['VAR'] if i.endswith("_Categorical")]
# filter unique categorical_variable_names
categorical_variable_names = list(set(categorical_variable_names))
categorical_variable_names_category = var_info[var_info['VAR'].isin(categorical_variable_names)]['VAR_CAT'].tolist()

#%%
'''

variable_bias tokenization

'''
# initialize a tokenizer with WordLevel model
tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token="[UNK]"))

# Then we know that BERT preprocesses texts by removing accents. We also use a unicode normalizer.
tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])

# The pre-tokenizer is just splitting on whitespace and punctuation:
tokenizer.pre_tokenizer = Whitespace()

# add special tokens
special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]
special_tokens = special_tokens
tokenizer.add_special_tokens(special_tokens)
#tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=512)

# Post-processing: add CLS and SEP.
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ("[UNK]", tokenizer.token_to_id("[UNK]")),
    ],
)

# Decoding strategy
tokenizer.decoder = decoders.WordPiece()

all_variable_names = categorical_variable_names + numerical_variable_names

# add snplist as new tokens to the tokenizer
tokenizer.add_tokens(all_variable_names)

ftt_tokenizer_dir = "model/ftt_variable_bias_tokenizer"
os.makedirs(ftt_tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(ftt_tokenizer_dir, "tokenizer.json"))

tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ftt_tokenizer_dir, "tokenizer.json"))
# specify special tokens for example [MASK] to mask_token
special_tokens_dict = {'mask_token': '[MASK]',
                       'pad_token': '[PAD]',
                       'cls_token': '[CLS]',
                       'sep_token': '[SEP]',
                       'unk_token': '[UNK]',
                       }
tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.save_pretrained(ftt_tokenizer_dir)

# %%
my_tokenizer = AutoTokenizer.from_pretrained(ftt_tokenizer_dir)
my_tokenizer.get_vocab()
# %%

'''

variable_weights tokenization

'''
# initialize a tokenizer with WordLevel model
tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token="[UNK]"))

# Then we know that BERT preprocesses texts by removing accents. We also use a unicode normalizer.
tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])

# The pre-tokenizer is just splitting on whitespace and punctuation:
tokenizer.pre_tokenizer = Whitespace()

# add special tokens
special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "[UNK]"]
special_tokens = special_tokens
tokenizer.add_special_tokens(special_tokens)
#tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=512)

# Post-processing: add CLS and SEP.
tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ("[UNK]", tokenizer.token_to_id("[UNK]")),
    ],
)

# Decoding strategy
tokenizer.decoder = decoders.WordPiece()

# set categorical and continuous variable names
all_variable_names = categorical_variable_names_category + numerical_variable_names

# add snplist as new tokens to the tokenizer
tokenizer.add_tokens(all_variable_names)

ftt_tokenizer_dir = "model/ftt_variable_weights_tokenizer"
os.makedirs(ftt_tokenizer_dir, exist_ok=True)
tokenizer.save(os.path.join(ftt_tokenizer_dir, "tokenizer.json"))

tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(ftt_tokenizer_dir, "tokenizer.json"))
# specify special tokens for example [MASK] to mask_token
special_tokens_dict = {'mask_token': '[MASK]',
                       'pad_token': '[PAD]',
                       'cls_token': '[CLS]',
                       'sep_token': '[SEP]',
                       'unk_token': '[UNK]',
                       }
tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.save_pretrained(ftt_tokenizer_dir)

# %%
my_tokenizer = AutoTokenizer.from_pretrained(ftt_tokenizer_dir)
my_tokenizer.get_vocab()
# %%
