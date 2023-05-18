#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pw
export CUDA_VISIBLE_DEVICES=2


dataset="seame"
model="tiny"
dataset_dir="path/to/seame/seame/data"
core_metric="mer"
split="valid" # for seame, it should be valid, devsge or devman, the later two are usually treated as test set in the literature
single_lang_threshold=1
concat_lang_token=1
code_switching="zh-en"
# need both concat_lang_token to be 1 and code_switching to be "zh-en" to enable lang concat in the prompt
# only turn code-switching to be "zh-en" will do normal whisper LID to select language token for the prompt
# if code-switching is "0", you should pass in a language token e.g. "zh", and we will therefore use this for all utterances
mkdir -p ./logs/${dataset}

echo "currently testing ${model}"
exp_name="${model}_${split}"
python ../csasr_st.py \
--data_split ${split} \
--single_lang_threshold ${single_lang_threshold} \
--concat_lang_token ${concat_lang_token} \
--code_switching ${code_switching} \
--model ${model} \
--dataset ${dataset} \
--dataset_dir ${dataset_dir} \
--core_metric ${core_metric} \
--beam_size 5 \
--topk 1000 \
--task transcribe 
# >> "./logs/${dataset}/${exp_name}.log" 2>&1