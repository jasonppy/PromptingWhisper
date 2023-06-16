#!/bin/sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pw
export CUDA_VISIBLE_DEVICES=2


dataset="visspeech"
model="medium.en"
dataset_dir="path/to/visspeech/data"
core_metric="wer"
pk="0"
ok="50"
num_img=3
socratic="1" # "1" mean also input visual prompt utilizing CLIP, "0" mean audio only

mkdir -p logs/${dataset}
echo "currently testing ${model} pk ${pk} ok ${ok}"
exp_name="${model}_placesk${pk}_objectk${ok}"
python ../avsr.py \
--place_topk $pk \
--obj_topk $ok \
--socratic $socratic \
--language "en" \
--num_img ${num_img} \
--model ${model} \
--dataset ${dataset} \
--dataset_dir ${dataset_dir} \
--core_metric ${core_metric} \
--batch_size 32 \
--beam_size 5 \
--topk 600 \
--task transcribe \
--object_txt_fn 'path/to/place_and_object/dictionary_and_semantic_hierarchy.txt' \
--place_txt_fn 'path/to/place_and_object/categories_places365.txt' \
--object_pkl_fn "path/to/place_and_object/tencent_336.pkl"
--place_pkl_fn "path/to/place_and_object/places365_336.pkl" >> "./logs/${dataset}/${exp_name}.log" 2>&1
