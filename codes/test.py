r1 = "police killed the gunman."
r2 = "the gunman was shot down by police."
c1 = "police ended the gunman."
c2 = "the gunman murdered police."

# using bert_score from datasets
from datasets import load_metric
# scorer = load_metric("./rouge.py")
scorer = load_metric("rouge")
s = scorer.compute(
    predictions=[c1, c2],
    references=[r1, r2],
    use_aggregator=False,
    use_stemmer=True,
)
print(s)

# import evaluate

# rouge = evaluate.load('rouge')
# predictions = ["hello there", "general kenobi"]
# references = ["hello there", "general kenobi"]
# results = rouge.compute(predictions=predictions, references=references)
# print(results)



# (base) [byron@spartan-login3 model_generation]$ sbatch a_gpt2_finetune_sep.slurm 
# Submitted batch job 41752928
# (base) [byron@spartan-login3 model_generation]$ sbatch a_gpt2_finetune_sep_chronological.slurm 
# Submitted batch job 41752929
# (base) [byron@spartan-login3 model_generation]$ sbatch a_gpt2_finetune_sep_mask_justices.slurm 
# Submitted batch job 41752930
# (base) [byron@spartan-login3 model_generation]$ sbatch a_gpt2_finetune_sep_mask_advocates.slurm 
# Submitted batch job 41752931
# (base) [byron@spartan-login3 model_generation]$ sbatch a_gpt2_finetune_sep_mask_justices_mask_advocates.slurm 
# Submitted batch job 41752933
# (base) [byron@spartan-login3 model_generation]$ sbatch a_gpt2_finetune_sep_chronological_mask_justices.slurm 
# Submitted batch job 41752934
# (base) [byron@spartan-login3 model_generation]$ sbatch a_gpt2_finetune_sep_chronological_mask_advocates.slurm 
# Submitted batch job 41752935
# (base) [byron@spartan-login3 model_generation]$ sbatch a_gpt2_finetune_sep_chronological_mask_justices_mask_advocates.slurm 
# Submitted batch job 41752938
# (base) [byron@spartan-login3 model_generation]$ 