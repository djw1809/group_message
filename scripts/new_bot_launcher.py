import pandas as pd
import numpy as np
import torch
import matplotlib
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import bot_utils as butils
import json
import bot_models as models
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path
import bot_models as models
#%%

model_dict = {'prepend': GPT2LMHeadModel, 'keyword': models.GPT2Model_bagofctrl}
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

raw_data_path = '../data/topics_index_bots_new_042820.pkl'

raw_data_text_field = 'tweet'
with open(raw_data_path, 'rb') as file:
    raw_data = pickle.load(file)

short_raw_data = {i:raw_data[i] for i in list(raw_data.keys())[0:7]}
short_raw_data[69] = {'tweet': 'I love dicks', 'topic_links': [], } #for testing

preprocessor = butils.Comment_data_preprocessor(raw_data, raw_data_text_field, tokenizer)
dataset1 = preprocessor.prepare_keyword_dataset(preprocessor.input_df, 'id', 'text', 'topic_links', key = 'types_nosentiment_nocluster', sentiment = False, cluster = False)
dataset2 = preprocessor.prepare_keyword_dataset(preprocessor.input_df, 'id', 'text', 'topic_links', key = 'types_sentiment_nocluster', sentiment = True, cluster = False)
dataset3 = preprocessor.prepare_keyword_dataset(preprocessor.input_df, 'id', 'text', 'topic_links', key = 'types_nosentiment_cluster', sentiment = False, cluster = True)
dataset4 = preprocessor.prepare_keyword_dataset(preprocessor.input_df, 'id', 'text', 'topic_links', key = 'types_sentiment_cluster', sentiment = True, cluster = True)

results_dir = '../results'
model_storage_dir = '../saved_models'
file_stem = 'batch_051220'

parameter_dict = {}
parameter_dict['epochs'] = 8
parameter_dict['num_worker'] = 2
parameter_dict['batch_size'] = 5
parameter_dict['learning_rate'] =5e-5
parameter_dict['weight_decay'] = 0
parameter_dict['eps'] =1e-8
parameter_dict['warmup_steps'] =0
parameter_dict['filename'] = ''




def train_batch_of_models(preprocessor, parameter_dict, results_dir = results_dir, model_storage_dir = model_storage_dir, type = 'keyword', file_stem = file_stem):
    datasets = preprocessor.prepared_datasets

    for dataset_name in datasets.keys():
        print("starting to train model on" + dataset_name)
        model = model_dict[type].from_pretrained('gpt2')
        preprocessor.set_active_dataset(dataset_name)

        parameter_dict['filename'] = file_stem + '_' + type + '_' + dataset_name
        results_path = Path(Path(results_dir)/Path(parameter_dict['filename']))
        model_path = Path(Path(model_storage_dir)/Path(parameter_dict['filename']))
        results_path.mkdir(parents = True, exist_ok = True)
        model_path.mkdir(parents = True, exist_ok = True)

        if type == 'keyword':
            preprocessor.set_get_type('keyword')
            trained_model, optimizer, scheduler, loss_data = butils.train_bag_of_words(preprocessor,
                                                                          parameter_dict['epochs'],
                                                                          parameter_dict['num_worker'],
                                                                          parameter_dict['batch_size'],
                                                                          parameter_dict['learning_rate'],
                                                                          parameter_dict['weight_decay'],
                                                                          parameter_dict['eps'],
                                                                          parameter_dict['warmup_steps'],
                                                                          model,
                                                                          collate_fn = preprocessor.collate_fn)
        if type == 'prepend':
            preprocessor.set_get_type('prepend_space')
            trained_model, optimizer, scheduler, loss_data = butils.train(dataset,
                                                                          parameter_dict['epochs'],
                                                                          parameter_dict['num_worker'],
                                                                          parameter_dict['batch_size'],
                                                                          parameter_dict['learning_rate'],
                                                                          parameter_dict['weight_decay'],
                                                                          parameter_dict['eps'],
                                                                          parameter_dict['warmup_steps'],
                                                                          model,
                                                                          collate_fn = preprocessor.collate_fn
                                                                          )
        #saving transformers stuff - this can all be loaded again using (transformer_object).from_pretrained(model_storage_dir+'/'+parameter_dict['filename'])
        preprocessor.active_dataset.to_csv(results_path/'training_data.csv')
        trained_model.save_pretrained(model_storage_dir + '/' + parameter_dict['filename'])
        tokenizer.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])
        trained_model.config.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])

        #saving torch stuff - see torch docs for proper loading
        torch.save(optimizer.state_dict(), Path(model_path)/Path(parameter_dict['filename']+' optimizer'))
        torch.save(scheduler.state_dict(), Path(model_path)/Path(parameter_dict['filename']+' scheduler'))

        #saving parameter dict
        with open(results_path/'parameters.json', 'w') as jsonFile:
            json.dump(parameter_dict, jsonFile)

        np.savetxt(results_path/'loss_data', loss_data, delimiter = ',')

        #plotting
        plt.clf()
        plt.scatter(range(parameter_dict['epochs']), loss_data)
        plt.savefig(results_dir + '/' + parameter_dict['filename'] +'/'+'loss_plot.png')


if __name__ == '__main__':
    train_batch_of_models(preprocessor, parameter_dict, type = 'keyword')
