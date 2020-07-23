import pandas as pd
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import messenger_utils as butils
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pathlib import Path
import messenger_models as models
# %%

tokenizer=GPT2Tokenizer.from_pretrained('gpt2')

training_set_path = '../data/datasets/cami_training.csv'
data = pd.read_csv(training_set_path)

#data = data[0:500]
parameter_dict = {}
#Currently huggingface defaults for training GPT2 (except more epochs)
parameter_dict['training_set_path'] = training_set_path
parameter_dict['epochs'] = 10
parameter_dict['num_worker'] = 2
parameter_dict['batch_size'] =2
parameter_dict['learning_rate'] =5e-5
parameter_dict['weight_decay'] = 0
parameter_dict['eps'] =1e-8
parameter_dict['warmup_steps'] =0
parameter_dict['filename'] = 'cami_bot_072320'

results_dir ='../results'
model_storage_dir ='../saved_models'

results_path = Path(Path(results_dir)/Path(parameter_dict['filename']))
model_path = Path(Path(model_storage_dir)/Path(parameter_dict['filename']))

results_path.mkdir(parents = True, exist_ok = True)
model_path.mkdir(parents = True, exist_ok = True)

dataset = butils.Comment_dataset(data, 'token_ids', tokenizer)

model = GPT2LMHeadModel.from_pretrained('gpt2')

trained_model, optimizer, scheduler, loss_data = butils.train(dataset,
                                                                  parameter_dict['epochs'],
                                                                  parameter_dict['num_worker'],
                                                                  parameter_dict['batch_size'],
                                                                  parameter_dict['learning_rate'],
                                                                  parameter_dict['weight_decay'],
                                                                  parameter_dict['eps'],
                                                                  parameter_dict['warmup_steps'],
                                                                  model,
                                                                  collate_fn = dataset.collate)



#saving transformers stuff - this can all be loaded again using (transformer_object).from_pretrained(model_storage_dir+'/'+parameter_dict['filename'])
data.to_csv(results_path/'training_data.csv')
trained_model.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])
tokenizer.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])
trained_model.config.save_pretrained(model_storage_dir+'/'+parameter_dict['filename'])

#saving torch stuff - see torch docs for proper loading
torch.save(optimizer.state_dict(), Path(model_path)/Path(parameter_dict['filename']+' optimizer'))
torch.save(scheduler.state_dict(), Path(model_path)/Path(parameter_dict['filename']+' scheduler'))

#saving parameter dict
with open(results_path/'parameters.json', 'w') as jsonFile:
    json.dump(parameter_dict, jsonFile)

#saving loss_data
np.savetxt(results_path/'loss_data', loss_data, delimiter = ',')

#plotting
plt.clf()
plt.scatter(range(parameter_dict['epochs']), loss_data)
plt.savefig(results_dir + '/' + parameter_dict['filename'] +'/'+'loss_plot.png')
