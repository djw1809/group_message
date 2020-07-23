import json
import pandas as pd
import torch
import bot_utils as butils
import bot_models as models
from transformers import GPT2Tokenizer

#%% all messages
big_data = []

for i in range(1,18):
    with open('../data/group_message/most_recent/message_{}.json'.format(i), 'rb') as file:
        data = json.load(file)
    data = data['messages']
    big_data = big_data + data
big_df = pd.DataFrame(big_data)
big_df

#%% camis messages
cami_df = big_df[big_df['sender_name'] == 'Cami Keyani']
cami_df = cami_df.dropna(subset = ['content'])
other_cami_df = cami_df[cami_df['type'] == 'Generic']
other_cami_df.to_csv('../data/datasets/cami_051320.csv')

#%% my messages
me_df = big_df[big_df['sender_name'] == 'Dylan Weber']
me_df = cami_df.dropna(subset = ['content'])
other_me_df = cami_df[cami_df['type'] == 'Generic']
other_me_df.to_csv('../data/datasets/dylan_051320.csv')

#%% wills messages
will_df = big_df[big_df['sender_name'] == 'Will Alberding']
will_df = cami_df.dropna(subset = ['content'])
other_will_df = cami_df[cami_df['type'] == 'Generic']
other_will_df.to_csv('../data/datasets/cami_051320.csv')

other_cami_df

#%% loading it into a preprocessor
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
preprocessor = butils.Comment_data_preprocessor(other_cami_df, 'content', tokenizer)
preprocessor.input_df['id'] = preprocessor.input_df.index
tokenized_df = preprocessor.df_to_tokenized_df(preprocessor.input_df)

for row in tokenized_df.index:
    if len(tokenized_df.loc[row, 'token_ids']) > 1023:
        tokenized_df.at[row, 'token_ids'] = tokenized_df.loc[row, 'token_ids'][:1023]

tokenized_df.to_csv('../data/datasets/cami_training.csv')

tokenized_df
