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

# import rouge

import numpy as np 
import json 
import os 
from itertools import chain
import logging
logging.basicConfig(level='ERROR')



set_seed(123) 
# set_seed(0)

# gpu = -1
# torch.cuda.set_device(gpu)


# dataset_name = "gpt2_legal_convos" 
dataset_name = sys.argv[1] 

# dataset_name = "legal_convos_mask_justices"

print("we use the dataset setting: {}".format(dataset_name))


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

# train_path = None 
# if "chronological" in dataset_name:  
#     if "half_dev" in dataset_name: 
#         train_path = "../generation_data/tagging_utterances_fillter_sides_train_2017_forward_and_half_2018.json" 
#     else: 
#         train_path = "../generation_data/tagging_utterances_fillter_sides_train_2017_forward.json"
# else: 
#     train_path = "../generation_data/tagging_utterances_fillter_sides_train_per_year.json"  

# print("train_path: ", train_path)

# original_train_dataset = load_dataset('json', data_files=train_path, split="train") 

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


# val_dataset_involved_justices = extract_justices_from_the_asking(original_val_dataset)
test_dataset_involved_justices = extract_justices_from_the_asking(original_test_dataset)


# print(len(test_dataset_involved_justices.keys()), test_dataset_involved_justices.keys())

# load model 


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




# the also load the finetuned parts 
model_path = 'log/{}'.format(dataset_name)    # it is where the best model is due to the trainer argument load_best_model_at_end = True
print("we are loading training checkpoint from: ", model_path) 

model = GPT2LMHeadModel.from_pretrained(model_path, config=config)


# training parameters



# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size. 

# https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
block_size = 1024
def group_texts_train(examples): # for training set
    print("-----------------------using concatenation approach------------------------------")
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



def group_texts_for_generation(examples): # for test set

    print("-----------------------group_texts_for_generation------------------------------")
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
        utterance_for_advocate_response += ar.split(" ")[0] + " " 

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
    # results["prompt_for_justice_asking"] = prompt_for_justice_asking 

    return results



def group_texts_for_counterfactual_generation(examples): # for test set

    print("-----------------------group_texts_for_counterfactual_generation------------------------------")
    # print('examples["justice_asking"]: ', examples["justice_asking"])
    prompt_for_advocate_response = []
    
    for pc, ja, ar in zip(examples["previous_context"], examples["justice_asking"], examples["advocate_response"]):  

        counterfactual_prompt_for_advocate_response = {}
        for replaced_justice_name in justices_tags:    

            # for avocate response 
            utterance_for_advocate_response = ja 
            # Counterfactual here!  - part 1 
            # replacing the last tag  - which is in ja 
            for t in justices_tags: 
                utterance_for_advocate_response = utterance_for_advocate_response.replace(t, replaced_justice_name)

            # utterance += " [ADV] " # adding the prompt, with adv hint  
            utterance_for_advocate_response += " " 
            utterance_for_advocate_response += ar.split(" ")[0] + " " 

            # adding previous context 
            
            # actually we need to mask pc too. so merge it together? 

            utterance_for_advocate_response = pc + " " + utterance_for_advocate_response
            
            if "mask_justices" in dataset_name:
                for t in justices_tags: 
                    utterance_for_advocate_response = utterance_for_advocate_response.replace(t, "[JST]")
            if "mask_advocates" in dataset_name:
                utterance_for_advocate_response = utterance_for_advocate_response.replace("[ADVPET]", "[ADV]").replace("[ADVRES]", "[ADV]").replace("[ADVUNSPE]", "[ADV]")
        


            # Counterfactual here!  - part 2
            # replacing the whole tags  or replace the last tag  
            # replacing the whole tags

            for t in justices_tags: 
                utterance_for_advocate_response = utterance_for_advocate_response.replace(t, replaced_justice_name)

            # replacing the last tag 
            # the "justice_name" is the original one - > find the last justice_name and replace it with the "replaced_justice_name"?  actually we can do it in ja before adding to previous context 

            counterfactual_prompt_for_advocate_response[replaced_justice_name] = utterance_for_advocate_response

        prompt_for_advocate_response.append(counterfactual_prompt_for_advocate_response) 

    results = {}
    results["prompt_for_advocate_response"] = prompt_for_advocate_response
    # results["prompt_for_justice_asking"] = prompt_for_justice_asking 

    return results



test_dataset = original_test_dataset.map(group_texts_for_counterfactual_generation, batched=True) 

def customized_truncation(input_ids, num_of_tokens): 
    # assuimg there is no padding 
    assert input_ids[-1]!= model.config.pad_token_id 

    # only get the last few tokens  
    input_ids = input_ids[-num_of_tokens: ] 
 
    return input_ids

def find_all(a_str, sub):  
    indexs = []
    start = 0
    while True:
        start = a_str.find(sub, start) 
        if start == -1: break
        indexs.append(start)
        start += len(sub) # use start += 1 to find overlapping matches 
    return indexs

def post_process_generated_text(genrated_text, prompt): 
    # make sure that it only has the immediate advocate response 
    special_tokens_indexs_generated_text = []
    special_tokens_indexs_prompt = []
    for token in special_tokens:  
        special_tokens_indexs_generated_text.extend(find_all(genrated_text, token)) 
        special_tokens_indexs_prompt.extend(find_all(prompt, token))

    special_tokens_indexs_generated_text = sorted(special_tokens_indexs_generated_text) 
    special_tokens_indexs_prompt = sorted(special_tokens_indexs_prompt)

    # now we use the spekcial token index in prompt to locate the text in generated_text 
    assert special_tokens_indexs_prompt == special_tokens_indexs_generated_text[: len(special_tokens_indexs_prompt)]

    separate_text = [] 
    for i in range(len(special_tokens_indexs_generated_text)): 
        if i == len(special_tokens_indexs_generated_text) -1: #last one 
            separate_text.append(genrated_text[special_tokens_indexs_generated_text[i]: ])
        else: 
            separate_text.append(genrated_text[special_tokens_indexs_generated_text[i]: special_tokens_indexs_generated_text[i+1]]) 
    # print("separated_text") 
    # print(separate_text) 
    selected_text = separate_text[len(special_tokens_indexs_prompt)-1] 

    # print("selected_text") 
    # print(selected_text) 
    # print()
    # assert selected_text.split(" ")[0] in ['[ADVRES]', '[ADVPET]', '[ADV]']  
    return selected_text

# print(test_dataset[0])

def get_gpt2_generation_result(model, input_ids, max_new_tokens): 
    model.eval() 
    beam_output = model.generate(
        input_ids, 
        # max_length=512, 
        max_new_tokens = max_new_tokens,  
        min_length=0, 
        num_beams=5, 
        # length_penalty = 2.0,
        early_stopping=True, 
        no_repeat_ngram_size = 3,
    ) 
    # print(len(beam_output)) 
    generated_text = tokenizer.decode(beam_output[0])

    # greedy_output = model.generate(input_ids, num_beams=1,  max_new_tokens = max_new_tokens)
    # generated_text = tokenizer.decode(greedy_output[0])
    # print("greedy! ")
    return generated_text
    

def remove_speaker_tags(predictions, references):
    for i in range(len(predictions)):  

        spk_gar = predictions[i].split(" ")[0].strip()
        spk_ar = references[i].split(" ")[0].strip()  

        # double check
        tmp_spk_ar = copy.deepcopy(spk_ar) 
        if "mask_justices" in dataset_name:
            for t in justices_tags: 
                tmp_spk_ar = tmp_spk_ar.replace(t, "[JST]")
        if "mask_advocates" in dataset_name: 
            tmp_spk_ar = tmp_spk_ar.replace("[ADVPET]", "[ADV]").replace("[ADVRES]", "[ADV]").replace("[ADVUNSPE]", "[ADV]")     

        # assert spk_gar == tmp_spk_ar

        if spk_gar !=  tmp_spk_ar: 
            print("-------------- speaker not matched ------------------") 
            print("prediction: ", predictions[i]) 
            print("reference: ", references[i])
        # print(spk_gar, spk_ar, tmp_spk_ar)
        predictions[i] = predictions[i].replace(spk_gar, "").strip() 
        references[i] = references[i].replace(spk_ar, "").strip()


def get_rouge_scores(predictions, references): 
    
    # print(predictions) 
    # print(references)
    rouge = load_metric('rouge')
    # predictions = ["hello there", "general kenobi"]
    # references = ["hello there", "general"]
    # rouge_results = rouge.compute(predictions=predictions, references=references)

    # print(rouge_results)
    # print(list(rouge_results.keys()))
    # print(rouge_results["rougeLsum"].mid.fmeasure) 

    # calculate the ave manually  
    rouge_results = rouge.compute(predictions=predictions, references=references, use_aggregator=False) 

    final_rouge_results = {} 
    for k in rouge_results.keys(): 
        final_rouge_results[k] = {"precision": [], "recall": [], "fmeasure": []}

        for rs in rouge_results[k]: 
            final_rouge_results[k]["precision"].append(rs.precision) 
            final_rouge_results[k]["recall"].append(rs.recall) 
            final_rouge_results[k]["fmeasure"].append(rs.fmeasure) 

    for k in final_rouge_results.keys(): 
        for m in final_rouge_results[k].keys(): 
            final_rouge_results[k][m] = sum(final_rouge_results[k][m])/len(final_rouge_results[k][m]) 
    return final_rouge_results



def get_bert_scores(predictions, references): 
    
    bertscore = load_metric("bertscore") 
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    # print(results)

    for k in results.keys(): 
        if k in ["precision", "recall", "f1"]:
            results[k] = sum(results[k])/len(results[k]) 
    
    return results



model.cuda() 



def generate_counterfactual_results_for_model(num_of_prompt_tokens): 
    print("-------------- with the prompt in {} tokens in counterfactual -------------------------".format(num_of_prompt_tokens))
    # predictions_advocate_response = {}
    # references_advocate_response = {}

    # separate into 5
    batch_gap = 4

    for p_index, p_text in enumerate(tqdm(test_dataset)):  
        if p_index % 5 == batch_gap: 

            p_text["generated_advocate_response"] = {}
            for replaced_justice_name in justices_tags:  

                max_new_tokens_for_advocate_response = 200
                # max_new_tokens_for_justice_asking = 100
                # for advocate response 

                for test_jst in justices_tags: 
                    if test_jst == replaced_justice_name: continue 
                    assert test_jst not in p_text["prompt_for_advocate_response"][replaced_justice_name]
                    
                input_ids_for_advocate_response = torch.tensor(tokenizer.encode(p_text["prompt_for_advocate_response"][replaced_justice_name])) 

                # directly getting input_ids as it only use the encoder - no attention mask here
                # print(input_ids.shape)
                # print(input_ids)
                # truncate the inpput_ids to the size we actally want -  only keey the last k tokens.
                input_ids_for_advocate_response = customized_truncation(input_ids_for_advocate_response, num_of_prompt_tokens)


                input_ids_for_advocate_response = input_ids_for_advocate_response.unsqueeze(0).cuda() 
                generated_text_for_advocate_response = get_gpt2_generation_result(model, input_ids_for_advocate_response, max_new_tokens_for_advocate_response)

                truncated_prompt_for_advocate_response = tokenizer.decode(input_ids_for_advocate_response[0]) 

                generated_text_for_advocate_response = post_process_generated_text(generated_text_for_advocate_response, truncated_prompt_for_advocate_response) 

                p_text["generated_advocate_response"][replaced_justice_name] = generated_text_for_advocate_response  

                # predictions_advocate_response[replaced_justice_name].append(generated_text_for_advocate_response) 
                # references_advocate_response[replaced_justice_name].append(p_text["advocate_response"])

            del p_text["prompt_for_advocate_response"]
                
            with open(os.path.join("generated_counterfactual_random_test", "{}_{}_{}_{}.json".format(dataset_name, num_of_prompt_tokens, max_new_tokens_for_advocate_response, p_index)), "w") as fw:  
                json.dump(p_text, fw)
                fw.write("\n")

    # remove_speaker_tags(predictions_advocate_response, references_advocate_response) 
    # remove_speaker_tags(predictions_justice_asking, references_justice_asking)


    # print("--------------- For advocates response-----------------------")
    # rouge_scores_for_advocate_response = get_rouge_scores(predictions_advocate_response, references_advocate_response) 
    # # print(rouge_scores_for_advocate_response)
    # rouge_score_store["advocate_response"][num_of_prompt_tokens] = rouge_scores_for_advocate_response

    # bert_scores_for_advocate_response = get_bert_scores(predictions_advocate_response, references_advocate_response) 
    # bert_score_store["advocate_response"][num_of_prompt_tokens] = bert_scores_for_advocate_response

  

# data_store_path = os.path.join("generated_text", dataset_name +".json")  



# with open(data_store_path, "w") as fw:  
# rouge_score_store = {"advocate_response": {}, "justice_asking": {}}
# bert_score_store = {"advocate_response": {}, "justice_asking": {}}

# rouge_score_store = {"advocate_response": {} }
# bert_score_store = {"advocate_response": {} }

# min_prompt_tokens = 0 
# max_prompt_tokens = 100
# gap = 25

# for num_of_prompt_tokens in range(min_prompt_tokens, max_prompt_tokens, gap):
#     if num_of_prompt_tokens == 0: num_of_prompt_tokens = 1

#     generate_results_for_model(num_of_prompt_tokens, rouge_score_store, bert_score_store)


# min_prompt_tokens = 100
# max_prompt_tokens = 900
# gap = 100

# for num_of_prompt_tokens in range(min_prompt_tokens, max_prompt_tokens, gap): 
#     if num_of_prompt_tokens == 0: num_of_prompt_tokens = 1

#     generate_results_for_model(num_of_prompt_tokens, rouge_score_store, bert_score_store)

min_prompt_tokens = 400
max_prompt_tokens = 401
gap = 100

for num_of_prompt_tokens in range(min_prompt_tokens, max_prompt_tokens, gap): 
    if num_of_prompt_tokens == 0: num_of_prompt_tokens = 1

    # generate_results_for_model(num_of_prompt_tokens, rouge_score_store, bert_score_store)
    generate_counterfactual_results_for_model(num_of_prompt_tokens)



# print("--------------- Final results -----------------------")
# print(rouge_score_store)
# print(bert_score_store)
# with open('check_results_{}.txt'.format(dataset_name), 'w') as fw:
#     fw.write("rouge scores:\n")
#     fw.write(json.dumps(rouge_score_store))  
#     fw.write("\nbert scores:\n") 
#     fw.write(json.dumps(bert_score_store))  



# from evaluate import load
# bertscore = load("bertscore")
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
# bertscore = load_metric("bertscore") 
# results = bertscore.compute(predictions=predictions, references=references, lang="en")
# print([round(v, 2) for v in results["f1"]])