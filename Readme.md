# It’s not only What You Say, It’s also Who It’s Said to: Counterfactual Analysis of Interactive Behavior in the Courtroom

## Introduction

This repository contains code introduced in the following paper:

- It’s not only What You Say, It’s also Who It’s Said to: Counterfactual analysis of Interactive Behavior in the Courtroom

- Biaoyan Fang, Trevor Cohn, Timothy Baldwin, and Lea Frermann 

- In IJCNLP-AACL 2023

## Dataset 

- This dataset a subset of [the Super-SCOTUS dataset](https://github.com/biaoyanf/Super-SCOTUS), a multi-sourced dataset for the Supreme Court of the US. We make this subset available under the directory `generation_data`, which we used for the experiment in this paper. 




## Training Instructions
- Run with python 3
- Run `python gpt2_finetune.py <experiment>`, where experiment can be `gpt2_legal_convos` and `gpt2_legal_convos_chronological` 


## Evaluation
- Evaluation: `python gpt2_evaluation_perplexity.py <experiment>`, where the experiment listed above 


## For the other work we have done 
-  
- 