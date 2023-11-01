from concurrent.futures import process
from curses import savetty
from macpath import join
import re
from tokenize import Special
from unittest import result
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments 
from transformers import set_seed
from datasets import load_dataset
from datasets import load_metric
from datasets import Dataset
import random
import sys
import torch

import copy 
from tqdm import tqdm

import numpy as np 
import json 
import os 
from itertools import chain
import logging
logging.basicConfig(level='ERROR')




set_seed(123) 

# gpu = -1
# torch.cuda.set_device(gpu)


# dataset_name = "gpt2_legal_convos" 
dataset_name = sys.argv[1] 

print("Now we are running setting: {}".format(dataset_name))
# load datasets

def load_jsonfile(file_path): 
    cases = [] 
    with open(file_path, "r") as fr: 
        for line in fr:
            cases.append(json.loads(line))  

    return cases

train_path = None 
if "chronological" in dataset_name:  
    train_path = "../generation_data/tagging_utterances_fillter_sides_train_2017_forward.json"
else: 
    train_path = "../generation_data/tagging_utterances_fillter_sides_train_per_year.json"  

print("train_path: ", train_path)

original_train_dataset = load_dataset('json', data_files=train_path, split="train") 
# train_dataset = load_jsonfile(train_path)
# print(type(train_dataset))
# print(train_dataset)
# train_dataset = Dataset.from_list(train_dataset)
# data_idx = random.choices(range(len(train_dataset)), k=24)
# train_dataset = train_dataset.select(data_idx)


# print(train_dataset)

# val_path = "../generation_data/val.csv" 
val_path = None 
if "chronological" in dataset_name: 
    val_path = "../generation_data/tagging_utterances_fillter_sides_dev_2018.json"  
else:
    val_path = "../generation_data/tagging_utterances_fillter_sides_dev_per_year.json" 

print("val_path: ", val_path)

original_val_dataset = load_dataset('json', data_files=val_path, split="train") 
# data_idx = random.choices(range(len(val_dataset)), k=16)
# val_dataset = val_dataset.select(data_idx)

test_path = None 
if "chronological" in dataset_name: 
    test_path = "../generation_data/tagging_utterances_fillter_sides_test_2019_pairs.json"
else:
    test_path = "../generation_data/tagging_utterances_fillter_sides_test_per_year_pairs.json" 

print("test_path: ", test_path)
original_test_dataset = load_dataset('json', data_files=test_path, split="train") 
# test_dataset = load_jsonfile(test_path)
# data_idx = random.choices(range(len(test_dataset)), k=8)
# test_dataset = test_dataset.select(data_idx) 

# test_dataset = [ele for i, ele in enumerate(test_dataset) if i in data_idx] 



# model_name = "facebook/bart-large-cnn" 
model_name = "gpt2"
# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# adding a lot of special tokens 

# print("before adding: ", len(tokenizer))

special_tokens =  ['[JSTBRETTMKAVANAUGH]', '[JSTWARRENEBURGER]', '[JSTFELIXFRANKFURTER]', '[JSTELENAKAGAN]', '[JSTANTONINSCALIA]', '[JSTLEWISFPOWELLJR]', '[JSTSTANLEYREED]', '[JSTTHURGOODMARSHALL]', '[JSTDAVIDHSOUTER]', '[JSTHAROLDBURTON]', '[JSTABEFORTAS]', '[JSTJOHNMHARLAN2]', '[JSTSAMUELAALITOJR]', '[JSTNEILGORSUCH]', '[JSTJOHNMHARLAN]', '[JSTCHARLESEWHITTAKER]', '[JSTEARLWARREN]', '[JSTANTHONYMKENNEDY]', '[JSTSONIASOTOMAYOR]', '[JSTSANDRADAYOCONNOR]', '[JSTJOHNPAULSTEVENS]', '[JSTRUTHBADERGINSBURG]', '[JSTWILLIAMODOUGLAS]', '[JSTPOTTERSTEWART]', '[JSTTOMCCLARK]', '[JSTWILLIAMHREHNQUIST]', '[JSTBYRONRWHITE]', '[JSTHUGOLBLACK]', '[JSTHARRYABLACKMUN]', '[JSTSTEPHENGBREYER]', '[JSTWILLIAMJBRENNANJR]', '[JSTCLARENCETHOMAS]', '[JSTJOHNGROBERTSJR]', '[JSTARTHURJGOLDBERG]', '[JSTSHERMANMINTON]', '[JST]', '[ADVRES]', '[ADVPET]', '[AMI]', '[ADVUNSPE]', '[ADV]', '[UNKNOWN]']
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

tokenizer.pad_token = tokenizer.eos_token

# print(len(special_tokens))
# print("after adding: ", len(tokenizer))

justices_tags = []  
for t in special_tokens: 
    if "JST" in t: 
        justices_tags.append(t)
# print(len(justices_tags), justices_tags)
# need to test if the tokenizer is actually works 

test_sentence = " I have dream adv ".join(special_tokens) 
# print(tokenizer(test_sentence))
test_tokens_ids = tokenizer(test_sentence) 
# print(test_tokens_ids)
# print(tokenizer.convert_ids_to_tokens(test_tokens_ids["input_ids"]))





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




# change that mapping too   
# see if it is actally the input 

# def convert_to_features(example_batch):
#     max_sourece, max_target = 64, 64 

#     input_encodings = tokenizer(example_batch['justice_speaking'], padding="max_length", max_length=max_sourece, return_tensors="pt", truncation=True)
#     target_encodings = tokenizer(example_batch['advocate_speaking'], padding="max_length", max_length=max_target, return_tensors="pt", truncation=True)

#     labels = target_encodings['input_ids']
#     labels[labels[:, :] == model.config.pad_token_id] = -100

#     encodings = {
#         'input_ids': input_encodings['input_ids'].detach().numpy(),
#         'attention_mask': input_encodings['attention_mask'].detach().numpy(),
#         'labels': labels.detach().numpy()   # the input should be the same input 
#     }
#     # print(encodings)
#     return encodings

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size. 

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
block_size = 1024
def group_texts_train(examples): # for training set
    print("-----------------------using concatenation approach------------------------------")

    if "mask_justices" in dataset_name:
        print("-----------------------mask justices------------------------------")
    if "mask_advocates" in dataset_name:
        print("-----------------------mask advocates------------------------------")
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

    # now it will all be ids 

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

def group_texts_train_sep(examples): # for training set 

    print("-----------------------using separation approach------------------------------") 

    if "mask_justices" in dataset_name:
        print("-----------------------mask justices------------------------------")
    if "mask_advocates" in dataset_name:
        print("-----------------------mask advocates------------------------------")
    
    # Concatenate all texts.
    # we should merge it in here 
    # examples = {k: list(chain(*tokenizer(examples[k]))) for k in examples.keys()}
    
    # it should the process text  
    # get_the_batch  
    results = {"input_ids": [], "attention_mask": []}
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
        

        # tmp_text has all the utterances for one conve/case  

        # so here we tokenize it 
        processed_examples = tokenizer(" ".join(tmp_text)) 
        # print("processed_examples: ", processed_examples.keys(), type(processed_examples["input_ids"]), len(processed_examples["input_ids"]))

        # now it will all be ids 

        # concatenated_examples = {k: list(chain(*processed_examples[k])) for k in processed_examples.keys()}
        concatenated_examples = processed_examples
        total_length = len(concatenated_examples[list(processed_examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        block_total_length = (total_length // block_size) * block_size

       
        # if block_total_length control the saving block 

        tmp_result = {
            k: [t[i : i + block_size] for i in range(0, block_total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        # if block_total_length !=0: continue 
        if total_length < block_size: 
            print("short text:", block_total_length)
            print(b) 
            print(tmp_result)
        # handle the rest 
        if block_total_length != total_length: 
            # process the remainder 
            # input_ids 
            remainder_input_ids = np.zeros(block_size, dtype=int) 
            assert len(concatenated_examples["input_ids"][block_total_length:] ) == total_length-block_total_length
            # if len(concatenated_examples["input_ids"][block_total_length:] ) != total_length-block_total_length: 
            #     print(len(concatenated_examples["input_ids"][block_total_length:]) , total_length-block_total_length, len(concatenated_examples["input_ids"]), total_length, block_total_length)
                # 0 -12119 169 169 12288
            remainder_input_ids[ :total_length-block_total_length] = concatenated_examples["input_ids"][block_total_length: ] 
            remainder_input_ids[total_length-block_total_length: ] = model.config.pad_token_id 
            # att mask
            remainder_att_mask = np.zeros(block_size, dtype=int) 
            remainder_att_mask[:total_length-block_total_length] = 1 

            tmp_result["input_ids"].append(remainder_input_ids) 
            tmp_result["attention_mask"].append(remainder_att_mask) 
        # print(len(tmp_result["input_ids"]) )
        # print(tmp_result["attention_mask"][-2])
        # add the tmp_result to results 
        results["input_ids"].extend(tmp_result["input_ids"]) 
        results["attention_mask"].extend(tmp_result["attention_mask"])
        # print("after adding: ", len(results["input_ids"]))

   
    results["labels"] = copy.deepcopy(results["input_ids"])
    for i in range(len(results["labels"])): 
        for j in range(len(results["labels"][i])): 
            if results["labels"][i][j] == model.config.pad_token_id: 
                results["labels"][i][j] = -100 

    return results



def group_texts_val(examples): # for test set
    # print('examples["justice_asking"]: ', examples["justice_asking"])
    print("-----------------------using group_texts val------------------------------")
    if "mask_justices" in dataset_name:
        print("-----------------------mask justices------------------------------")
    if "mask_advocates" in dataset_name:
        print("-----------------------mask advocates------------------------------") 

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
    results = tokenizer(processed_text, padding="max_length", max_length=max_length, truncation=True) 

    # result["labels"] = result["input_ids"].copy()
    # need to do a deepcopy and set the padding part as -100 for cross entropy calculation 
    results["labels"] = copy.deepcopy(results["input_ids"])

    for i in range(len(results["labels"])): 
        for j in range(len(results["labels"][i])): 
            if results["labels"][i][j] == model.config.pad_token_id: 
                results["labels"][i][j] = -100 

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
#     result = tokenizer(processed_text, padding="max_length", max_length=512) 



#     # result["labels"] = result["input_ids"].copy()
#     # need to do a deepcopy and set the padding part as -100 for cross entropy calculation 
#     result["labels"] = copy.deepcopy(result["input_ids"])

#     for i in range(len(result["labels"])): 
#         for j in range(len(result["labels"][i])): 
#             if result["labels"][i][j] == model.config.pad_token_id: 
#                 result["labels"][i][j] = -100 

#     return result




batch_size = 2

# cnn_train = cnn_train.map(convert_to_features, batched=True)
# cnn_val = cnn_val.map(convert_to_features, batched=True)
# cnn_test = cnn_test.map(convert_to_features, batched=True)
# cnn_train.set_format(type='torch', columns=columns)
# cnn_val.set_format(type='torch', columns=columns)
# cnn_test.set_format(type='torch', columns=columns)

# print(train_dataset)
# train_dataset = train_dataset.map(convert_to_features, batched=True,batch_size=batch_size)
# val_dataset = val_dataset.map(convert_to_features, batched=True, batch_size=batch_size) 
# test_dataset = test_dataset.map(convert_to_features, batched=True, batch_size=batch_size) 
if "sep" in dataset_name: 
    train_dataset = original_train_dataset.map(group_texts_train_sep, batched=True, remove_columns=["convos", "id"])
    val_dataset = original_val_dataset.map(group_texts_train_sep, batched=True, remove_columns=["convos", "id"]) 
else: 
    train_dataset = original_train_dataset.map(group_texts_train, batched=True, remove_columns=["convos", "id"])
    val_dataset = original_val_dataset.map(group_texts_train, batched=True, remove_columns=["convos", "id"]) 

# val_dataset = val_dataset.map(group_texts_val, batched=True) 
test_dataset = original_test_dataset.map(group_texts_val, batched=True) 

columns = ['input_ids', 'labels', 'attention_mask']
train_dataset.set_format(type='torch', columns=columns)
val_dataset.set_format(type='torch', columns=columns)
test_dataset.set_format(type='torch', columns=columns)

# print(train_dataset) 
# print("__________________________________") 
# print(test_dataset)

# print(len(train_dataset["input_ids"][0]), len(train_dataset["input_ids"][-1])) 

training_args = Seq2SeqTrainingArguments(
    do_train=True,
    do_eval=True,
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
    # report_to='wandb',
    # no_cuda = True,
    load_best_model_at_end=True,
    # metric_for_best_model = "mask_condition_loss", 
    # greater_is_better = True,
    num_train_epochs=50
)


# compute Rouge score during validation
# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions

#     pred_strs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True) 
#     # pred_strs = tokenizer.batch_decode(pred_ids) 
#     # pred_strs = [p.replace("<|endoftext|>", "") for p in pred_strs]
#     # labels_ids[labels_ids == -100] = tokenizer.pad_token_id
#     label_strs = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
#     # label_strs = tokenizer.batch_decode(labels_ids)
#     # label_strs = [l.replace("<|endoftext|>", "") for l in label_strs]
#     # print(len(pred_strs[0]), pred_strs[0][:10]) 
    
#     # print("pred_strs: ", pred_strs) 
#     # print("label_strs: ", label_strs)
#     # print("_______________________________________")
#     rouge_scorer = load_metric('rouge')
#     rouge_output = rouge_scorer.compute(
#         predictions=pred_strs,
#         references=label_strs,
#         use_aggregator=False,
#         use_stemmer=True,
#     )
#     # print(rouge_output["rouge1"].fmeasure)  
#     result = {} 
#     for t in rouge_output.keys(): 
#         result[t] = {"precision": 0, "recall": 0, "fmeasure": 0}
#         for ele in rouge_output[t]: 
#             result[t]["precision"] += ele.precision 
#             result[t]["recall"] += ele.recall  
#             result[t]["fmeasure"] += ele.fmeasure
#         for k in result[t]: 
#             result[t][k] /= len(rouge_output[t]) 
         
#     result_dict = {
#         "rouge1_fmeasure": round(result["rouge1"]["fmeasure"], 4),
#         "rouge2_fmeasure": round(result["rouge2"]["fmeasure"], 4),
#         "rougeL_fmeasure": round(result["rougeL"]["fmeasure"], 4),
#         "rougeLsum_fmeasure": round(result["rougeLsum"]["fmeasure"], 4),
#     }
#     # print(result_dict)
#     return result_dict

def compute_metrics_perplexity(pred): 
    original_label_ids = pred.label_ids   # so it will the same as input but with paddings set as -100 now 
    original_logits = pred.predictions 

    # now the logits and labels_ids are numpy.array
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
    # compute_metrics=compute_metrics_perplexity,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# test_results = trainer.predict(test_dataset=test_dataset)
# # print("Get the result hey ")
# print(test_results.metrics) 
# print("done with naive prediction")
# start training
trainer.train()

# test_results = trainer.predict(test_dataset=test_dataset)
# print(test_results.metrics)

save_path = "./log/{}".format(dataset_name)
print("Save model to {}".format(save_path)) 

model.save_pretrained(save_path)