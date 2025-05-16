#!/bin/bash
clear
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$1

method=mbr3
# embed, mbr3, rerank_mbr3
# model_name=llama2-7b-chat
# model_name=llama3-8b-instruct
model_name=llama3.1-8b-instruct
# llama2-7b-chat, llama3-8b-instruct, llama3.1-8b-instruct

topk=8
min_length=$2
max_length=$3
num_per=16
num_test=5
temperature=0.4

run_dir=output/niah
# fname=mbr3
fname=original
mkdir -p $run_dir/$model_name/$fname
pred_path=${run_dir}/${model_name}/${fname}/pred_min${min_length}_max${max_length}.jsonl
# pred_path=${run_dir}/${model_name}/${fname}/pred_chunk_num${chunk_num}_min${min_length}_max${max_length}_topk${topk}.json
echo "pred_path: ${pred_path}"

# greedy
if [ ! -e ${pred_path} ]; then
  python inference_niah.py \
    hydra.run.dir=${run_dir}/${model_name}/${fname} \
    model_name=${model_name} \
    output_file=${pred_path} \
    task_name=${task_name} \
    topk=${topk} \
    method=${method} \
    max_length=${max_length} \
    min_length=${min_length} \
    num_per=${num_per} \
    num_test=1 \
    generation_kwargs.do_sample=False
fi 

# sample
# if [ ! -e ${pred_path} ]; then
#   python inference_niah.py \
#     hydra.run.dir=${run_dir}/${model_name}/${fname} \
#     model_name=${model_name} \
#     output_file=${pred_path} \
#     task_name=${task_name} \
#     topk=${topk} \
#     method=${method} \
#     max_length=${max_length} \
#     min_length=${min_length} \
#     num_per=${num_per} \
#     generation_kwargs.do_sample=True \
#     generation_kwargs.temperature=${temperature} \
#     generation_kwargs.num_return_sequences=${num_test} 
# fi 
