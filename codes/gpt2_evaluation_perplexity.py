from cProfile import label
from concurrent.futures import process
from ctypes import sizeof
from curses import savetty
from email.policy import default
from functools import total_ordering
# from lib2to3.pytree import _Results
from macpath import join
from multiprocessing import reduction
import re
from tokenize import Special
from unittest import result
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments 
from transformers import set_seed
from datasets import load_dataset
from datasets import load_metric
from datasets import Dataset 
from torch.utils.data import DataLoader

import random
import sys
import torch
from sklearn.metrics import log_loss 
from tqdm import tqdm


import math
import numpy as np 
import json 
import os 
import copy
from itertools import chain
import logging 
logging.basicConfig(level='ERROR')



set_seed(123)  

# load datasets
# dataset_name = "legal_convos" 
dataset_name = sys.argv[1] 

# dataset_name = "legal_convos_mask_justices"

print("we use the dataset setting: {}".format(dataset_name))

def load_jsonfile(file_path): 
    cases = [] 
    with open(file_path, "r") as fr: 
        for line in fr:
            cases.append(json.loads(line))  

    return cases

def extract_justices_from_the_asking(extract_dataset): 
    extract_justices = {} 
    for i, ele in enumerate(extract_dataset): 
        justice_name = ele["justice_asking"].split(" ")[0]
        if justice_name not in extract_justices: extract_justices[justice_name] = [] 
        # eval_justices.add(justice_name)  
        extract_justices[justice_name].append(i)  
    # assert sum(ex)
    assert sum([len(extract_justices[k]) for k in extract_justices]) == len(extract_dataset) 
    
    return extract_justices


train_path = "../generation_data/tagging_utterances_fillter_sides_train_per_year.json" 
train_dataset = load_dataset('json', data_files=train_path, split="train") 
# train_dataset = load_jsonfile(train_path)
# print(type(train_dataset))
# print(train_dataset)
# train_dataset = Dataset.from_list(train_dataset)
data_idx = random.choices(range(len(train_dataset)), k=8)
train_dataset = train_dataset.select(data_idx)


# train_dataset = [ele for i, ele in enumerate(train_dataset) if i in data_idx] 


# print(train_dataset)

# val_path = "../generation_data/val.csv" 
val_path = None 
if "chronological" in dataset_name: 
    val_path = "../generation_data/tagging_utterances_fillter_sides_dev_2018_pairs.json"  
else:
    val_path = "../generation_data/tagging_utterances_fillter_sides_dev_per_year_pairs.json" 

val_dataset = load_dataset('json', data_files=val_path, split="train") 
# data_idx = random.choices(range(len(val_dataset)), k=16)
# data_idx = [1]*2 +[2]*4
# val_dataset = val_dataset.select(data_idx)

test_path = None 
if "chronological" in dataset_name: 
    test_path = "../generation_data/tagging_utterances_fillter_sides_test_2019_pairs.json"
else:
    test_path = "../generation_data/tagging_utterances_fillter_sides_test_per_year_pairs.json" 

test_dataset = load_dataset('json', data_files=test_path, split="train") 
# test_dataset = load_jsonfile(test_path)
# data_idx = random.choices(range(len(test_dataset)), k=8)
# test_dataset = test_dataset.select(data_idx) 

# test_dataset = [ele for i, ele in enumerate(test_dataset) if i in data_idx] 

val_dataset_involved_justices = extract_justices_from_the_asking(val_dataset)
test_dataset_involved_justices = extract_justices_from_the_asking(test_dataset)


# print(len(test_dataset_involved_justices.keys()), test_dataset_involved_justices.keys())

model_name = "gpt2"
# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# adding a lot of special tokens 

# print("before adding: ", len(tokenizer))

special_tokens =  ['[JSTBRETTMKAVANAUGH]', '[JSTWARRENEBURGER]', '[JSTFELIXFRANKFURTER]', '[JSTELENAKAGAN]', '[JSTANTONINSCALIA]', '[JSTLEWISFPOWELLJR]', '[JSTSTANLEYREED]', '[JSTTHURGOODMARSHALL]', '[JSTDAVIDHSOUTER]', '[JSTHAROLDBURTON]', '[JSTABEFORTAS]', '[JSTJOHNMHARLAN2]', '[JSTSAMUELAALITOJR]', '[JSTNEILGORSUCH]', '[JSTJOHNMHARLAN]', '[JSTCHARLESEWHITTAKER]', '[JSTEARLWARREN]', '[JSTANTHONYMKENNEDY]', '[JSTSONIASOTOMAYOR]', '[JSTSANDRADAYOCONNOR]', '[JSTJOHNPAULSTEVENS]', '[JSTRUTHBADERGINSBURG]', '[JSTWILLIAMODOUGLAS]', '[JSTPOTTERSTEWART]', '[JSTTOMCCLARK]', '[JSTWILLIAMHREHNQUIST]', '[JSTBYRONRWHITE]', '[JSTHUGOLBLACK]', '[JSTHARRYABLACKMUN]', '[JSTSTEPHENGBREYER]', '[JSTWILLIAMJBRENNANJR]', '[JSTCLARENCETHOMAS]', '[JSTJOHNGROBERTSJR]', '[JSTARTHURJGOLDBERG]', '[JSTSHERMANMINTON]', '[JST]', '[ADVRES]', '[ADVPET]', '[AMI]', '[ADVUNSPE]', '[ADV]', '[UNKNOWN]']
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

tokenizer.pad_token = tokenizer.eos_token



justices_tags = []  
for t in special_tokens: 
    if "JST" in t: 
        justices_tags.append(t)

# test_sentence = " I have dream adv ".join(special_tokens) 
# # print(tokenizer(test_sentence))
# test_tokens_ids = tokenizer(test_sentence) 



config = GPT2Config.from_pretrained(model_name)
# print(config)
config.gradient_checkpointing = False
# set generate hyperparameters
config.num_beams = 5
config.max_length = 1024
config.min_length = 0
config.length_penalty = 2.0
config.early_stopping = True
config.no_repeat_ngram_size = 3
config.pad_token_id = tokenizer.eos_token_id 

model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
# need to resize as we add special tokens
model.resize_token_embeddings(len(tokenizer))

# training parameters



block_size = 1024
def group_texts_train(examples): # for training set
    # Concatenate all texts.
    # we should merge it in here 
    # examples = {k: list(chain(*tokenizer(examples[k]))) for k in examples.keys()}
    
    # it should the process text  
    # get_the_batch 
    processed_text = []
    for b in examples["convos"]: 
        tmp_text = []
        for u in b:  
            utterance = u
            # masking the side of advocates 
            if "mask_advocates" in dataset_name:
                utterance = utterance.replace("[ADVPET]", "[ADV]").replace("[ADVRES]", "[ADV]").replace("[ADVUNSPE]", "[ADV]")
            # masking the actual name of justcie 
            if "mask_justices" in dataset_name:
                for t in justices_tags: 
                    utterance = utterance.replace(t, "[JST]")
            tmp_text.append(utterance)
        
            
        processed_text.append(" ".join(tmp_text)) 

    # print("processed_text in train: ", processed_text)
    
    processed_examples = tokenizer(processed_text) 
    # print("processed_examples: ",  len(processed_examples["input_ids"]), len(processed_examples["input_ids"][0]))


    concatenated_examples = {k: list(chain(*processed_examples[k])) for k in processed_examples.keys()}
    total_length = len(concatenated_examples[list(processed_examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # print("result: ",  len(result["input_ids"]), len(result["input_ids"][0])) 
    # print("result: ",  len(result["attention_mask"]), len(result["attention_mask"][0])) 
    # print("result: ", result.keys())
    result["labels"] = result["input_ids"].copy()
    return result

def group_texts_val(examples): # for test set
    # print('examples["justice_asking"]: ', examples["justice_asking"])
    processed_text = []
    for ja, ar in zip(examples["justice_asking"], examples["advocate_response"]): 
        utterance = ja 
        if "mask_justices" in dataset_name:
                for t in justices_tags: 
                    utterance = utterance.replace(t, "[JST]")

        # utterance += " [ADV] " # adding the prompt, with adv hint  
        utterance += " " 
        utterance += ar
        if "mask_advocates" in dataset_name:
            utterance = utterance.replace("[ADVPET]", "[ADV]").replace("[ADVRES]", "[ADV]").replace("[ADVUNSPE]", "[ADV]")

        processed_text.append(utterance) 

    # print("processed_text in dev/test: ", processed_text)
    max_length = 512 
    result = tokenizer(processed_text, padding="max_length", max_length=max_length, truncation=True) 

    # result["labels"] = result["input_ids"].copy()
    # need to do a deepcopy and set the padding part as -100 for cross entropy calculation 
    result["labels"] = copy.deepcopy(result["input_ids"])

    for i in range(len(result["labels"])): 
        for j in range(len(result["labels"][i])): 
            if result["labels"][i][j] == model.config.pad_token_id: 
                result["labels"][i][j] = -100 

    return result


def group_texts_val_with_previous_context(examples): # for test set

    print("-----------------------group_texts_val_with_previous_context------------------------------")
    # print('examples["justice_asking"]: ', examples["justice_asking"])
    prompt_for_advocate_response = []
    prompt_for_justice_asking = []
    for pc, ja, ar in zip(examples["previous_context"], examples["justice_asking"], examples["advocate_response"]):  

        # for avocate response 
        utterance_for_advocate_response = ja 
        # if "mask_justices" in dataset_name:
        #         for t in justices_tags: 
        #             utterance_for_advocate_response = utterance_for_advocate_response.replace(t, "[JST]")

        # utterance += " [ADV] " # adding the prompt, with adv hint  
        utterance_for_advocate_response += " " 
        # utterance_for_advocate_response += ar.split(" ")[0] + " " 

        # adding previous context 
        
        # actually we need to mask pc too. so merge it together? 

        utterance_for_advocate_response = pc + " " + utterance_for_advocate_response
        
        if "mask_justices" in dataset_name:
            for t in justices_tags: 
                utterance_for_advocate_response = utterance_for_advocate_response.replace(t, "[JST]")
        if "mask_advocates" in dataset_name:
            utterance_for_advocate_response = utterance_for_advocate_response.replace("[ADVPET]", "[ADV]").replace("[ADVRES]", "[ADV]").replace("[ADVUNSPE]", "[ADV]")
        prompt_for_advocate_response.append(utterance_for_advocate_response) 


        # # for justice asking 
        # utterance_for_justice_asking = ja.split(" ")[0] + " "  
        
        # utterance_for_justice_asking = pc + " " + utterance_for_justice_asking
        # if "mask_justices" in dataset_name:
        #     for t in justices_tags: 
        #         utterance_for_justice_asking = utterance_for_justice_asking.replace(t, "[JST]")
        # if "mask_advocates" in dataset_name:
        #     utterance_for_justice_asking = utterance_for_justice_asking.replace("[ADVPET]", "[ADV]").replace("[ADVRES]", "[ADV]").replace("[ADVUNSPE]", "[ADV]")

        # prompt_for_justice_asking.append(utterance_for_justice_asking)

    results = {}
    results["prompt_for_advocate_response"] = prompt_for_advocate_response

    return results




# def group_texts_test(examples): # for test set
#     # print('examples["justice_asking"]: ', examples["justice_asking"])
#     processed_text = []
#     for b in examples["justice_asking"]: 
#         utterance = b 
#         # for t in justices_tags: 
#         #     utterance = utterance.replace(t, "[JST]")

#         utterance += " [ADV] " # adding the prompt, with adv hint  
    
#         processed_text.append(utterance) 
#     # print(processed_text)
#     result = tokenizer(processed_text, padding="max_length", max_length=512, truncation=True) 



#     result["labels"] = result["input_ids"].copy()
#     return result



batch_size = 2


train_dataset = train_dataset.map(group_texts_train, batched=True, remove_columns=["convos", "id"])
val_dataset = val_dataset.map(group_texts_val_with_previous_context, batched=True) 
test_dataset = test_dataset.map(group_texts_val_with_previous_context, batched=True) 

# columns = ['input_ids', 'labels', 'attention_mask']
# train_dataset.set_format(type='torch', columns=columns)
# val_dataset.set_format(type='torch', columns=columns)
# test_dataset.set_format(type='torch', columns=columns)


training_args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
    # include_inputs_for_metrics = True, # version is too old and there is no such function yet
    # evaluate_during_training =False,
    do_predict=False,
    # predict_with_generate=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    output_dir="result_%s"%dataset_name,
    logging_strategy="steps",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    warmup_steps=1000,
    save_total_limit=2,
    gradient_accumulation_steps=8,
    eval_accumulation_steps = 1,
    # report_to='wandb',
    # no_cuda = True,
    # load_best_model_at_end=True,
    load_best_model_at_end=True,
    # metric_for_best_model = "mask_condition_loss", 
    # greater_is_better = True,
    num_train_epochs=50
)



def compute_metrics_perplexity(pred): 
    # so we will get the mask for some of the loss 

    # get the part that is for generation and then use it for the evaluation -  so it will be in the part after the first [ADV] token. 

    original_label_ids = pred.label_ids   # so it will the same as input but with paddings set as -100 now 
    original_logits = pred.predictions

    # # something for test  
    # logits = torch.from_numpy(logits)
    # shift_logits = logits[..., :-1, :].contiguous()
    # print(logits.size())
    # print(shift_logits.size()) 
    # # print(shift_logits)
    # print(shift_logits.view(-1, shift_logits.size(-1)).size())

    # labels = torch.from_numpy(label_ids)  
    # shift_labels = labels[..., 1:].contiguous()
    # print(labels.size())
    # print(shift_labels.size()) 
    # # print(shift_labels) 
    # print(shift_labels.view(-1).size()) 
    # modified_loss = torch.nn.CrossEntropyLoss()  
    # print(modified_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))

    # # done with testing 


    # assuming pred is the [bath, sequence, vocab] 

    # print(logits.shape) 
    # print(labels_ids.shape) 
    # print(labels_ids[1])

    # so it means I might need to shift the labels too? 

    # we also shift the output here - this one is very important! 
    # print(type(logits), type(label_ids))
    # shift_logits = original_logits 
    # shift_label_ids =  original_label_ids
    
    shift_logits = original_logits[..., :-1, :]
    shift_label_ids = original_label_ids[..., 1:]  

    shift_mask_condition_label_ids = copy.deepcopy(shift_label_ids)  

    adv_id = tokenizer.encode("[ADV]") 
    advres_id = tokenizer.encode("[ADVRES]") 
    advpet_id = tokenizer.encode("[ADVPET]")  

    assert len(adv_id) == 1 
    assert len(advres_id) == 1 
    assert len(advpet_id) == 1  
    adv_id = adv_id[0]
    advres_id = advres_id[0] 
    advpet_id = advpet_id[0]

    # print(adv_id)
    for i in range(len(shift_mask_condition_label_ids)): 
        use_part = False  
        for j in range(len(shift_mask_condition_label_ids[i])): 
            shift_mask_condition_label_ids[i][j] = shift_mask_condition_label_ids[i][j] if use_part else -100 

            if shift_label_ids[i][j] == adv_id or shift_label_ids[i][j] == advres_id or shift_label_ids[i][j] == advpet_id: 
                use_part = True 

                # print("we do something! ")
    # print(use_label_ids[1])

    # modified_loss = torch.nn.CrossEntropyLoss(reduction = "none") 
    # check_loss = modified_loss(torch.from_numpy(logits).permute(0,2,1), torch.from_numpy(label_ids))
    # print(check_loss.size())
    # print(check_loss[0])

    # modified_loss = torch.nn.CrossEntropyLoss()  
    
    # result = modified_loss(logits, use_label_ids)
    # print()
    # result["loss_mask"] = [] 
    
    # for bath_input_ids in result["input_ids"]: 
    #     tmp_loss_mask = [] 
    #     use_part = 0 
    #     for input_id in bath_input_ids:
    #         tmp_loss_mask.append(use_part) 
    #         if input_id == adv_id: 
    #             use_part = 1 
    #             # print("we find the flip point! ") 
    #         # if input_id == model.config.pad_token_id:  # the padding 
    #         #     use_part = 0 # flip back! 
    #             # print("and here !!!!!!!!")

    #     result["loss_mask"].append(tmp_loss_mask)
    # check_loss = modified_loss(torch.from_numpy(logits).permute(0,2,1), torch.from_numpy(label_ids))
    # print(check_loss.size()  )
    # print(check_loss[0])


    # use the log and see how it goes  

    # hmm the loss seems different in  sklearn and pytorch

    # ---------------------------------------sklearn---------------------------------------------
    # loss_count = [] 
    # provide_all_labels = [i for i in range(len(tokenizer))] 

    # for y_pred, y_label in zip(shift_logits, shift_label_ids): 
    #     # extract the actual y_pred and y_label -> removinig all the elements as -100 
    #     extract_y_pred, extract_y_label = [], [] 
    #     for p, l in zip(y_pred, y_label): 
    #         if l == -100: 
    #             # print("hmmm")
    #             continue 
    #         extract_y_pred.append(p)
    #         extract_y_label.append(l)
    #     print(extract_y_label)
    #     loss_count.append(log_loss(extract_y_label, extract_y_pred,labels = provide_all_labels ))

    # print("we use sklearn ")
    # print(loss_count) 
    # print(sum(loss_count)/len(loss_count))

    #-------------------------------------tensor----------------------------------------------------
    # # need to process it via batch  and it might be too dum puting it to the tensor again? 

    # we need to do the batch thing again cuz here logits : [all_test_instances=6k+, seq=512, vob =5k+] too big to feed as only once. 
    # so using the batch parameter we have in the training

    print("are we here? ")   
    print(len(shift_label_ids))
    loss_ce = torch.nn.CrossEntropyLoss(reduction = "sum") 
    total_eval_instances = len(shift_label_ids)
    batch_eval_loss = {"mask_condition_loss": [], "non_mask_condition_loss": [], "non_mask_condition_loss_count": [],  "mask_condition_loss_count": []} 
    for i in range(0, total_eval_instances, batch_size): 
        tmp_shift_logits = shift_logits[i:i+batch_size, :, :] 
        tmp_shift_label_ids = shift_label_ids[i:i+batch_size, :] 
        tmp_shift_mask_condition_label_ids = shift_mask_condition_label_ids[i:i+batch_size, :] 
        with torch.no_grad():
            tmp_shift_logits = torch.from_numpy(tmp_shift_logits).contiguous()
            tmp_shift_label_ids = torch.from_numpy(tmp_shift_label_ids).contiguous()
            tmp_shift_mask_condition_label_ids = torch.from_numpy(tmp_shift_mask_condition_label_ids).contiguous()

            batch_eval_loss["non_mask_condition_loss"].append(loss_ce(tmp_shift_logits.view(-1, tmp_shift_logits.size(-1)), tmp_shift_label_ids.view(-1)))

            batch_eval_loss["mask_condition_loss"].append(loss_ce(tmp_shift_logits.view(-1, tmp_shift_logits.size(-1)), tmp_shift_mask_condition_label_ids.view(-1)))


            batch_eval_loss["non_mask_condition_loss_count"].append((tmp_shift_label_ids.view(-1)!=-100).sum())

            batch_eval_loss["mask_condition_loss_count"].append((tmp_shift_mask_condition_label_ids.view(-1)!=-100).sum())
    
    results = {}
    
    results["non_mask_condition_loss"] = sum(batch_eval_loss["non_mask_condition_loss"])/sum(batch_eval_loss["non_mask_condition_loss_count"])
    results["mask_condition_loss"] = sum(batch_eval_loss["mask_condition_loss"])/sum(batch_eval_loss["mask_condition_loss_count"])
    results["non_mask_condition_perplexity"] = torch.exp(results["non_mask_condition_loss"])
    results["mask_condition_perplexity"] = torch.exp(results["mask_condition_loss"])
    
    for k in results: 
        results[k] = results[k].item()
    # results["loss"] = 0
    # print(results)
    return results

   


# we need to set seed first 


# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    # compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)



model_path = 'log/{}'.format(dataset_name)    # it is where the best model is due to the trainer argument load_best_model_at_end = True

print("we are loading training checkpoint from: ", model_path) 

model = GPT2LMHeadModel.from_pretrained(model_path, config=config)


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics_perplexity,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# test_results = trainer.predict(test_dataset=val_dataset)
# print(test_results.metrics) 

# val_dataset = DataLoader(val_dataset, batch_size=batch_size)
# test_results = trainer.prediction_loop(val_dataset, "test")
# print(test_results.metrics)   

num_of_prompt_tokens = 400 
num_of_advocate_response = 512

def process_per_batch_result(logits, labels, raw_input): 
    shift_logits = logits[..., :-1, :].contiguous() 
    shift_labels = labels[..., 1:].contiguous() 

    #  do something with mask 
    shift_mask_condition_labels = shift_labels.detach().clone().contiguous() 

    adv_id = tokenizer.encode("[ADV]") 
    advres_id = tokenizer.encode("[ADVRES]") 
    advpet_id = tokenizer.encode("[ADVPET]")  
    
    # bos_id =  tokenizer.encode(tokenizer.bos_token)


    assert len(adv_id) == 1 
    assert len(advres_id) == 1 
    assert len(advpet_id) == 1  
    
    # assert len(bos_id) == 1 


    adv_id = adv_id[0]
    advres_id = advres_id[0] 
    advpet_id = advpet_id[0]

    # bos_id = bos_id[0]
    # print(adv_id)

    for i in range(len(shift_mask_condition_labels)): 

        for j in range(num_of_prompt_tokens +1):  #+1 cuz the first adv is the identifier  
            shift_mask_condition_labels[i][j] =  -100 

            

    loss_ce = torch.nn.CrossEntropyLoss(reduction = "sum")  
    # print(loss_cr(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))) 

    per_eval_loss = {"non_mask_condition_loss": loss_ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)), 
                    "non_mask_condition_loss_count": (shift_labels.view(-1)!=-100).sum(), 
                    "mask_condition_loss": loss_ce(shift_logits.view(-1, shift_logits.size(-1)), shift_mask_condition_labels.view(-1)), 
                    "mask_condition_loss_count": (shift_mask_condition_labels.view(-1)!=-100).sum()}   
    # print(per_eval_loss)
    return per_eval_loss

def customized_truncation(input_ids, num_of_tokens): 
    # assuimg there is no padding 
    assert input_ids[-1]!= model.config.pad_token_id 

    # only get the last few tokens  
    input_ids = input_ids[-num_of_tokens: ] 
    
    # if it is less 
    
    if len(input_ids) < num_of_tokens: 
        new_input_ids = [model.config.pad_token_id]* (num_of_tokens-len(input_ids) )
        new_input_ids.extend(input_ids) 
        input_ids = new_input_ids

    assert len(input_ids) == num_of_tokens

    return input_ids



def covert_raw_input_to_inputs(raw_input):  

    # we have 

    # for tag in special_tokens: 
    #     raw_input["prompt_for_advocate_response"] = raw_input["prompt_for_advocate_response"].replace(tag, tokenizer.bos_token)
    input_ids_for_advocate_response = [] 
    for e in raw_input["prompt_for_advocate_response"]: 
        tmp_ids = tokenizer.encode(e)  
        tmp_ids = customized_truncation(tmp_ids, num_of_prompt_tokens)
        input_ids_for_advocate_response.append(tmp_ids) 
    # import pdb
    # pdb.set_trace()

    # print(raw_input["prompt_for_advocate_response"])
    # print(input_ids_for_advocate_response) 
    # print("--------------------------------------------\n")
    # directly getting input_ids as it only use the encoder - no attention mask here
    # print(input_ids.shape)
    # print(input_ids)
    # truncate the inpput_ids to the size we actally want -  only keey the last k tokens.
    

    
    # input_ids_for_advocate_response = input_ids_for_advocate_response.unsqueeze(0).cuda() 
    generated_text_for_advocate_response = []
    for e in raw_input["advocate_response"]: 
        tmp_ids = tokenizer.encode(e, padding  = 'max_length', truncation = True, max_length = num_of_advocate_response, add_special_tokens = True) 

        generated_text_for_advocate_response.append(tmp_ids)

    # import pdb
    # pdb.set_trace()
    # print(raw_input["advocate_response"], generated_text_for_advocate_response) 
    # print(input_ids_for_advocate_response)
    # print("-----------------------------------")
    input_ids_for_advocate_response = torch.tensor(input_ids_for_advocate_response)
    generated_text_for_advocate_response = torch.tensor(generated_text_for_advocate_response)
    # print(input_ids_for_advocate_response.size(), generated_text_for_advocate_response.size())
    # so it will be 400 + 512 


    inputs_ids =  torch.cat((input_ids_for_advocate_response, generated_text_for_advocate_response), 1) 
    # print(inputs_ids.size())
    # inputs_ids =  input_ids_for_advocate_response + generated_text_for_advocate_response 
    
    attention_mask = [] 
    for i in range(len(inputs_ids)): 
        tmp_mask = [] 
        for j in range(len(inputs_ids[i])): 

            tmp_mask.append(int(inputs_ids[i][j]!=model.config.pad_token_id))  # 1 means use and 0 means not
        attention_mask.append(tmp_mask) 
    

    labels = inputs_ids.clone().detach()
    for i in range(len(inputs_ids)): 

        for j in range(len(inputs_ids[i])): 

            labels[i][j] = inputs_ids[i][j] if inputs_ids[i][j]!=model.config.pad_token_id else -100  



    results = {}
    results["input_ids"] = torch.tensor(inputs_ids)
    results["attention_mask"] = torch.tensor(attention_mask)
    results["labels"] = torch.tensor(labels)

    return results
    

    # truncated_prompt_for_advocate_response = tokenizer.decode(input_ids_for_advocate_response[0]) 


    # # print("processed_text in dev/test: ", processed_text)
    # max_length = 512 
    # result = tokenizer(processed_text, padding="max_length", max_length=max_length, truncation=True) 

    # # result["labels"] = result["input_ids"].copy()
    # # need to do a deepcopy and set the padding part as -100 for cross entropy calculation 
    # result["labels"] = copy.deepcopy(result["input_ids"])

    # for i in range(len(result["labels"])): 
    #     for j in range(len(result["labels"][i])): 
    #         if result["labels"][i][j] == model.config.pad_token_id: 
    #             result["labels"][i][j] = -100 


def evaluation_on_perplexity(eval_dataset): 
    eval_dataset = DataLoader(eval_dataset, batch_size=batch_size)  
    batch_eval_loss = {"loss": [], "non_mask_condition_loss": [], "mask_condition_loss": [], "non_mask_condition_loss_count": [],  "mask_condition_loss_count": []}
    for step, raw_input in enumerate(tqdm(eval_dataset)):

        inputs = covert_raw_input_to_inputs(raw_input)

        loss, logits, labels = trainer.prediction_step(model, inputs, prediction_loss_only = False) 
        # so those are tensors I assume 
        # print("loss:", loss) 
        # print("logits: ", logits.size())  

        # break
        batch_eval_loss["loss"].append(loss) 
        per_eval_loss = process_per_batch_result(logits, labels, raw_input) 
        for k in per_eval_loss: 
            batch_eval_loss[k].append(per_eval_loss[k])  


    results = {}
    results["loss"] = sum(batch_eval_loss["loss"])/len(batch_eval_loss["loss"]) 
    results["non_mask_condition_loss"] = sum(batch_eval_loss["non_mask_condition_loss"])/sum(batch_eval_loss["non_mask_condition_loss_count"]) 
    results["mask_condition_loss"] = sum(batch_eval_loss["mask_condition_loss"])/sum(batch_eval_loss["mask_condition_loss_count"])  
    results["non_mask_condition_perplexity"] = torch.exp(results["non_mask_condition_loss"])
    results["mask_condition_perplexity"] = torch.exp(results["mask_condition_loss"]) 


    # results["test_non_mask_condition_perplexity"] = torch.exp(torch.stack(batch_eval_loss["non_mask_condition_loss"])).sum() /len(batch_eval_loss["non_mask_condition_loss"])
    # results["test_mask_condition_perplexity"] = torch.exp(torch.stack(batch_eval_loss["mask_condition_loss"]) ).sum() /len(batch_eval_loss["mask_condition_loss"])

    # print(results)
    return results


# dict_keys(['previous_context', 'justice_asking', 'advocate_response', 'advocate_side', 'advocate_win', 'advocate_direction', 'justice_gender', 'justice_nomonated_party', 'justice_is_chief', 'justice_against_advocate']) 

# need to do something about the separation 
# dataset.filter(lambda example: example['sentence1'].startswith('Ar')) 

stored_results = {}

protected_attributes = ['advocate_side', 'advocate_win', 'advocate_direction', 'justice_gender', 'justice_nomonated_party', 'justice_is_chief', 'justice_against_advocate']


eval_dataset = test_dataset 
eval_dataset_involved_justices = test_dataset_involved_justices

print(eval_dataset)

print("__________________________________________________________________")
print("Overall results: ")

results = evaluation_on_perplexity(eval_dataset)
print(results) 

stored_results["overall"] = (len(eval_dataset), results["mask_condition_perplexity"])
print("overall {} instances with mask confition perplexity: {:.2f} ".format(len(eval_dataset), results["mask_condition_perplexity"]))
print("__________________________________________________________________")

