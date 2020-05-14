# %%
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from ast import literal_eval



class Comment_data_preprocessor(Dataset):
    '''class to do different things quickly with raw_data
        -properties: raw_data, input_df, train_df (the dataframe used for training), test_df(the dataframe used for testing), current_df(either test or train, the df that will be accessed by get_item), tokenizer, corpus
        - always assume input has text field, id field
        - TODO: assert that if a synonym dict has been provided then so should a keyword_field
        '''
    #PREPROCESSING
    def __init__(self, raw_data, text_field, tokenizer, keyword_field = None, synonym_dict = None):
        #sefl.tokenizer = tokenizer
        self.tokenizer = tokenizer
        self.synonym_dict = synonym_dict
        self.raw_data = raw_data
        self.corpus = None
        self.input_df = None
        self.prepared_datasets = {}
        self.active_dataset = None # This is the dataset that will be accessed by __getitem__ and __len__
        self.get_type = 'keyword'  #keyword: return a tuple of (tokenized keywords, tokenized text), prepend: return tokenized keyword prepended text
        self.collate_fn = self.collate_keyword

        if type(raw_data) == str: #process input data if it is a corpus
            self.corpus = raw_data
            self.input_df = self.text_to_chunked_df(self.raw_text)

        elif isinstance(raw_data, pd.DataFrame): #process input data if it is a dataframe
            self.input_df = raw_data
            self.input_df = self.input_df.rename(columns = {text_field: 'text'})
            # intermediate_df = pd.DataFrame(columns = ['id', 'text'])
            # intermediate_df.loc[:, 'id'] = raw_data.loc[:, id_field]
            # intermediate_df.loc[:, 'text'] = raw_data.loc[:, text_field]
            # if keyword_field != None:
            #     intermediate_df.loc[:, 'keywords'] = raw_data.loc[:, keyword_field]
            #     intermediate_df.loc[:, 'keywords'] = intermediate_df.loc[:, 'keywords'].apply(literal_eval) #make sure keywords are in a list


        elif isinstance(raw_data, list):  #process input data if it is json list of dicts, each dict representing a comment
            self.input_df = pd.DataFrame(raw_data)
            self.input_df = self.input_df.rename(columns = {text_field: 'text'})

            # if keyword_field != None:
            #     drop_columns = [column for column in intermediate_df.columns if column not in [id_field, text_field, keyword_field]]
            # else:
            #     drop_columns = [column for column in intermediate_df.columns if column not in [id_field, text_field]]
            #
            # intermediate_df = intermediate_df.drop(columns = drop_columns)
            #
            # if keyword_field != None:
            #     intermediate_df = intermediate_df.rename(columns = {id_field: 'id', text_field: 'text', keyword_field: 'keywords'})
            #     intermediate_df.loc[:, 'keywords'] = intermediate_df.loc[:, 'keywords'].apply(literal_eval) #make sure keywords are in a list
            # else:
            #     intermediate_df = intermediate_df.rename(columns = {id_field: 'id', text_field: 'text'})

        elif isinstance(raw_data, dict): #process input data if it is json dict of dicts, each dict representing a comment with ids as keys
            self.input_df = pd.DataFrame.from_dict(raw_data, orient = 'index')
            self.input_df['id'] = self.input_df.index
            self.input_df = self.input_df.rename(columns = {text_field: 'text'})


        else:
            pass

        self.input_df = self.input_df.dropna(subset = ['text']) #drop rows where there is no text
        self.input_df.index = range(len(self.input_df))

    def prepare_keyword_dataset(self, input_data, id_field, text_field, topic_link_field, key = None, sentiment = False, cluster = False):
        '''prepares dataframe with different ways of handling keywords.
           input: a dataframe with a field for text_ids, texts, and a topic link field. A topic link is a list of tuples of the form (cluster, phrase, phrase_synonym, disease, type).  Each text has a topic link for each phrase present in text. Keywords for a text always at least include the type.
           sentiment: include sentiment symbol in type keywords
           cluster: include cluster names in keywords
           output:
           if prepend: a dataframe with id, and prepend_text fields.  prepend_text is original text with keywords prepended.
           otherwise: a dataframe with id, text and keyword fields'''

        output_data = pd.DataFrame(columns = ['id', 'text', 'keywords'])

        for entry in input_data.index:
            output_data.loc[len(output_data) + 1] = np.nan
            id = input_data.loc[entry, id_field]
            text = input_data.loc[entry, text_field]
            topic_links = input_data.loc[entry, 'topic_links']

            if sentiment:
                type_keywords = [j[4] for j in topic_links if len(j[4]) > 0]
            else:
                type_keywords = [j[4].rstrip('+-') for j in topic_links if len(j[4]) > 0]

            if cluster:
                cluster_keywords = [j[0] for j in topic_links if len(j[0]) > 0]
                type_keywords = list(set(type_keywords))
                cluster_keywords = list(set(cluster_keywords))
                keywords = type_keywords + cluster_keywords
            else:
                type_keywords = list(set(type_keywords))
                keywords = type_keywords

            output_data.loc[output_data.index.max()] = {'id': id, 'text':text, 'keywords':keywords}



        output_data = self.df_to_tokenized_df(output_data)
        output_data.index = range(len(output_data))

        if key == None:
            key = len(self.prepared_datasets + 1)

        self.prepared_datasets[key] = output_data


        return output_data

    def df_to_tokenized_df(self, input_data, number_of_keywords = None):

        if 'keywords' in input_data.columns:
            tokenized_df = pd.DataFrame(columns = ['id', 'text', 'keywords', 'used_keywords', 'tokenized_text', 'tokenized_keywords', 'text_ids', 'keyword_ids'])
            tokenized_df.loc[:,'id'] = input_data.loc[:,'id']
            tokenized_df.loc[:,'text'] = input_data.loc[:,'text']
            tokenized_df.loc[:, 'tokenized_text'] = tokenized_df.loc[:, 'text'].apply(self.tokenizer.tokenize)
            tokenized_df.loc[:, 'text_ids'] = tokenized_df.loc[:, 'text'].apply(self.tokenizer.encode)
            tokenized_df.loc[:,'keywords'] = input_data.loc[:,'keywords']
            tokenized_df.index = input_data.index
            tokenized_df.loc[:,'used_keywords'] = [[] for i in tokenized_df.index]
            tokenized_df['used_keywords'] = tokenized_df.loc[:,'used_keywords'].astype('object')

            if self.synonym_dict != None:

                for row in tokenized_df.index:

                    keywords = input_data.loc[row, 'keywords'].copy()
                    translated_keywords = []
#
                    if number_of_keywords != None:
                        translate_range = number_of_keywords
                    else: #if no number is given translate all of them
                        translate_range = len(keywords)

                    for i in range(translate_range):
                        if len(keywords) == 0: #incase there are examples where the number of keywords is less than the translate range
                            break
                        else:
                            keyword = keywords.pop()
                            try:
                                translated_keywords.append(self.synonym_dict[keyword])
                            except KeyError: ##if there is no translation for the keyword just keep it
                                translated_keywords.append(keyword)

                    translated_keywords = list(set(translated_keywords)) #remove duplicates
                    tokenized_df.at[row, 'used_keywords'] = translated_keywords






            else:
                for row in tokenized_df.index:
                    keywords = list(input_data.loc[row, 'keywords'])
                    prepended_keywords = []
                    if number_of_keywords != None:
                        translate_range = number_of_keywords
                    else:
                        translate_range = len(keywords)
                        for i in range(translate_range):
                            if len(keywords) == 0:
                                break
                            else:
                                keyword = keywords.pop()
                                prepended_keywords.append(keyword)
                                tokenized_df.at[row, 'used_keywords'] = prepended_keywords



            tokenized_df.loc[:, 'tokenized_keywords'] = tokenized_df.loc[:, 'used_keywords'].apply(self.tokenize_list_of_keywords)
            tokenized_df.loc[:, 'keyword_ids'] = tokenized_df.loc[:, 'used_keywords'].apply(self.encode_list_of_keywords)


        else:
            tokenized_df = pd.DataFrame(columns = ['id', 'text', 'tokenized_text', 'token_ids'])
            tokenized_df.loc[:,'id'] = input_data.loc[:, 'id']
            tokenized_df.loc[:, 'text'] = input_data.loc[:, 'text']
            tokenized_df.loc[:, 'tokenized_text'] = input_data.loc[:, 'text'].apply(self.tokenizer.tokenize)
            tokenized_df.loc[:, 'token_ids'] = input_data.loc[:, 'text'].apply(self.tokenizer.encode)

        return tokenized_df


    def input_df_to_corpus(self):
        self.corpus = ''
        for i in range(len(self.input_df)):
            self.corpus = self.corpus + ' ' + self.input_df.loc[i, 'text']

    def tokenize_list_of_keywords(self, input_list):
        output_tokens = []
        for keyword in input_list:
            output_tokens = output_tokens + self.tokenizer.tokenize(keyword)

        return output_tokens

    def encode_list_of_keywords(self, input_list):
        output_ids = []
        for keyword in input_list:
            output_ids = output_ids + self.tokenizer.encode(keyword)

        return output_ids

    ##LOADING
    def set_active_dataset(self, key):
        self.active_dataset = self.prepared_datasets[key]

    def set_get_type(self, type):
        self.get_type = type
        if type == 'keyword':
            self.collate_fn = self.collate_keyword

        if type.startswith('prepend'):
            self.collate_fn = self.collate_prepend


    def __getitem__(self, index):
        try:
            if self.active_dataset == None:
                print("No active dataset")
                return
        except ValueError:
            pass

        if self.get_type == 'prepend_nospace':
            text_ids = self.active_dataset.loc[index, 'text_ids']
            keyword_ids = self.active_dataset.loc[index, 'keyword_ids']
            prepended_ids = keyword_ids + text_ids
            return prepended_ids

        if self.get_type == 'prepend_space':
            text = self.active_dataset.loc[index, 'text']
            keywords = self.active_dataset.loc[index,'keywords']
            prepended_text = text
            for keyword in keywords:
                prepended_text = keyword + ' ' + prepended_text
            prepended_ids = self.tokenizer.encode(prepended_text)
            return prepended_ids

        if self.get_type == 'keyword':
            text_ids = self.active_dataset.loc[index, 'text_ids']
            keyword_ids = self.active_dataset.loc[index, 'keyword_ids']
            return (text_ids, keyword_ids)

    def __len__(self):
        try:
            if self.active_dataset == None:
                print("No active dataset")
                return
        except ValueError:
            pass

        return len(self.active_dataset)

    def collate_prepend(self, batch):
        tokenizer = self.tokenizer
        text_ids = [torch.tensor(item) for item in batch]

        if tokenizer._pad_token is None:
             padded_texts = pad_sequence(text_ids, batch_first = True)
        else:
             padded_texts = pad_sequence(text_ids, batch_first = True, padding_value = tokenizer.pad_token_id)

        return padded_texts

    def collate_keyword(self, batch):
        tokenizer = self.tokenizer
        text_ids = [torch.tensor(item[0]) for item in batch]
        keyword_ids = [torch.tensor(item[1]) for item in batch]

        if tokenizer._pad_token is None:
             padded_texts = pad_sequence(text_ids, batch_first = True)
        else:
             padded_texts = pad_sequence(text_ids, batch_first = True, padding_value = tokenizer.pad_token_id)

        return padded_texts, keyword_ids









class Comment_dataset(Dataset):

    def __init__(self, raw_data, sample_column):
        self.sample_column = sample_column
        self.data = raw_data

    def __getitem__(self, index):
        return torch.tensor(eval(self.data.loc[index, self.sample_column]))

    def __len__(self):
        return len(self.data)

    def collate(self, batch):
        tokenizer = self.tokenizer
        text_ids = [torch.tensor(item) for item in batch]

        if tokenizer._pad_token is None:
             padded_texts = pad_sequence(text_ids, batch_first = True)
        else:
             padded_texts = pad_sequence(text_ids, batch_first = True, padding_value = tokenizer.pad_token_id)

        return padded_texts


class prepend_ctrl_Dataset(Dataset):

    def __init__(self, preprocessor, tokenizer = None):
        if preprocessor.tokenized_df is None:
            preprocessor.tokenized_df()

        self.data = preprocessor.tokenized_df

        if tokenizer != None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = preprocessor.tokenizer
            except:
                print("no tokenizer found - collate function wont work")


    def __getitem__(self, index):
        text_ids = self.data.loc[index, 'text_ids']
        keyword_ids = self.data.loc[index,'keyword_ids']
        prepended_ids = keyword_ids + text_ids
        return prepended_ids

    def __len__(self):
        return len(self.data)


    def collate(self, batch):
        tokenizer = self.tokenizer
        text_ids = [torch.tensor(item) for item in batch]

        if tokenizer._pad_token is None:
             padded_texts = pad_sequence(text_ids, batch_first = True)
        else:
             padded_texts = pad_sequence(text_ids, batch_first = True, padding_value = tokenizer.pad_token_id)

        return padded_texts


class bag_words_ctrl_Dataset(Dataset):

    def __init__(self, preprocessor, tokenizer = None):
        if preprocessor.tokenized_df is None:
            preprocessor.tokenized_df()

        self.data = preprocessor.tokenized_df
        if tokenizer != None:
            self.tokenizer = tokenizer
        else:
            try:
                self.tokenizer = preprocessor.tokenizer
            except:
                print("no tokenizer found - collate function wont work")

    def __getitem__(self, index):
        text_ids = self.data.loc[index, 'text_ids']
        keyword_ids = self.data.loc[index, 'keyword_ids']
        return (text_ids, keyword_ids)

    def __len__(self):
        return len(self.data)


    def collate(self, batch):
        tokenizer = self.tokenizer
        text_ids = [torch.tensor(item[0]) for item in batch]
        keyword_ids = [torch.tensor(item[1]) for item in batch]

        if tokenizer._pad_token is None:
             padded_texts = pad_sequence(text_ids, batch_first = True)
        else:
             padded_texts = pad_sequence(text_ids, batch_first = True, padding_value = tokenizer.pad_token_id)

        return padded_texts, keyword_ids










# %%


def train(training_dataset, epochs, num_workers, batch_size, learning_rate, weight_decay, eps, warmup_steps, model, collate_fn = None):
    '''generic training call for a pytorch model'''


    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size, collate_fn = collate_fn)


#### configure model to use cuda if it is available ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()

#### initialize containers to store model outputs in ####
    loss_data = np.zeros((epochs)) #empty arrays to store data for plotting in

### initialize optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ] #set weight decay to 0 for bias and layernorm weights

    optimizer = AdamW(optimizer_grouped_parameters, lr= learning_rate, eps= eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps= len(training_loader)
    )

##### MAIN TRAINING LOOP ######

    for epoch in range(epochs):

        running_loss = 0
        model.train()

        for batch  in training_loader:
            inputs, labels = (batch, batch)
            inputs = inputs.to(device)
            labels = labels.to(device)
            #optimizer.zero_grad()

            #forward
            loss = model(inputs, labels = labels)[0]

            #backwards
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            running_loss += loss.item()
            #running_corrects += torch.sum(preds == labels.data).item()
            #confusion_matrix_train_epoch += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels =range(num_labels))

##### calculate  epoch data ####
        epoch_loss = running_loss / len(training_dataset)
        #epoch_corrects = running_corrects / len(training_dataset)
        #epoch_val_accuracy = running_val_corrects/len(test_dataset)

###### record epoch data ###########
        loss_data[epoch] = epoch_loss
        #accuracy_data[epoch] = epoch_corrects
        #val_accuracy_data[epoch] = epoch_val_accuracy
        #confusion_matricies_test[epoch] = confusion_matrix_test_epoch
        #confusion_matricies_train[epoch] = confusion_matrix_train_epoch

        print(' Loss: {:.4f} '.format(epoch_loss))

    return model, optimizer, scheduler, loss_data


def train_bag_of_words(training_dataset, epochs, num_workers, batch_size, learning_rate, weight_decay, eps, warmup_steps, model, collate_fn = None):
    '''generic training call for a pytorch model'''


    training_loader = DataLoader(training_dataset, shuffle = True, num_workers = num_workers, batch_size = batch_size, collate_fn = collate_fn)


#### configure model to use cuda if it is available ####
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()

#### initialize containers to store model outputs in ####
    loss_data = np.zeros((epochs)) #empty arrays to store data for plotting in

### initialize optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ] #set weight decay to 0 for bias and layernorm weights

    optimizer = AdamW(optimizer_grouped_parameters, lr= learning_rate, eps= eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps= len(training_loader)
    )

##### MAIN TRAINING LOOP ######

    for epoch in range(epochs):

        running_loss = 0
        model.train()

        for batch  in training_loader:
            inputs, labels = (batch, batch[0])
            device_sequence = inputs[0].to(device)
            device_keywords = [inputs[1][i].to(device) for i in range(len(inputs[1]))]
            inputs = (device_sequence, device_keywords)
            labels = labels.to(device)
            #optimizer.zero_grad()

            #forward
            loss = model(inputs, device, labels = labels)[0]

            #backwards
            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            running_loss += loss.item()
            #running_corrects += torch.sum(preds == labels.data).item()
            #confusion_matrix_train_epoch += confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy(), labels =range(num_labels))

##### calculate  epoch data ####
        epoch_loss = running_loss / len(training_dataset)
        #epoch_corrects = running_corrects / len(training_dataset)
        #epoch_val_accuracy = running_val_corrects/len(test_dataset)

###### record epoch data ###########
        loss_data[epoch] = epoch_loss
        #accuracy_data[epoch] = epoch_corrects
        #val_accuracy_data[epoch] = epoch_val_accuracy
        #confusion_matricies_test[epoch] = confusion_matrix_test_epoch
        #confusion_matricies_train[epoch] = confusion_matrix_train_epoch

        print(' Loss: {:.4f} '.format(epoch_loss))

    return model, optimizer, scheduler, loss_data

# %%


def generate_(model, tokenizer, prompt, max_length, do_sample = True, num_beams = None, temperature = None, top_k = None, top_p = None, repetition_penalty = None, num_return_sequences = 1,   print_ = True, stop_token = None):
    '''generate with transformer models'''
    encoded_prompt = tokenizer.encode(prompt, add_special_tokens = False, return_tensors = "pt")
    output_sequences = model.generate(input_ids = encoded_prompt,
                                      max_length = max_length,
                                      temperature = temperature,
                                      top_k = top_k,
                                      top_p = top_p,
                                      repetition_penalty = repetition_penalty,
                                      do_sample = True,
                                      num_return_sequences = num_return_sequences)

    if len(output_sequences.shape) > 2:
        output_sequences = output_sequences.squeeze()

    generated_sequences = []

    for id, sequence in enumerate(output_sequences):
        decoded_sequence = tokenizer.decode(sequence.tolist(), clean_up_tokenization_spaces = True)
        if stop_token != None:
            decoded_sequence = decoded_sequence[: (decoded_sequence.find(stop_token) + 1)]

        if print_:
            print("Generated sequence {}: {}".format(id, prompt + ' ' + decoded_sequence))

        generated_sequences.append(prompt + ' ' + decoded_sequence)


    return output_sequences


def generate_ctrl_bagofwords(model, tokenizer, prompt, max_length, temperature = None, top_k = None, top_p = None, num_return_sequences = 1, print_ = True, min_keep = 1, filter_value = -float("Inf")):
    '''generation with bag of words ctrl.  prompt should be of the form (list of keywords, start of generated sentence)'''
    #setup device
    device = "cpu" #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #if torch.cuda.is_available():
        #model.cuda()
    model.eval()

    #encode prompt
    if len(prompt) == 1:
        keywords,  = prompt

    else:
        keywords, bos = prompt
        bos_tokens = tokenizer.encode(bos)

    keyword_tokens = []
    if len(keywords) > 0:
        for keyword in keywords:
            keyword_tokens = keyword_tokens + tokenizer.encode(keywords)

        keyword_tokens = [torch.tensor(keyword_tokens)] #put things in right shape for forward pass
    else:
        keyword_tokens = [[]]

    returned_sequences = []

    for i in range(num_return_sequences):
        if len(prompt) > 1:
            sequence_tokens = bos_tokens
        else:
            sequence_tokens = []
        for j in range(max_length):
            #obtain logits
            input_ids = torch.tensor(sequence_tokens).unsqueeze(0)
            logits = model((input_ids, keyword_tokens), device)[0][:, -1, :]

        #perform top_k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:,-1, None] #return the indicies
                logits[indices_to_remove] = filter_value  #mask the bad ones
        #perform "nucleus" sampling
            if top_p > 0:
                sorted_logits, sorted_indices = torch.sort(logits, descending = True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1 ), dim = -1)
                sorted_indices_to_remove = cum_probs > top_p
                if min_keep > 1:
                    sorted_indices_to_remove[:, :min_keep] = 0

                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone() #shift everything to right - will always pick first token above threshhold as well now
                sorted_indices_to_remove[:, 0] = 0  #always keep at least most probable

                #put everything in the right place
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)

                logits[indices_to_remove] = filter_value

            if top_k > 0 or top_p > 0:
                next_token_index = [int(torch.multinomial(F.softmax(logits), 1))]
            else:
                next_token_index = [int(torch.argmax(logits))]

            sequence_tokens = sequence_tokens + next_token_index

        returned_sequences.append(sequence_tokens)

    returned_sentences = []

    for sequence in returned_sequences:
        decoded_sequence = tokenizer.decode(sequence, clean_up_tokenization_spaces = True)
        returned_sentences.append(decoded_sequence)

    return returned_sentences, returned_sequences
