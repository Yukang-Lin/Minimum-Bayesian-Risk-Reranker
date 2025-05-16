#!/bin/bash
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

method=mbr3
# embed, mbr3, rerank_mbr3
layer=10
# model_name=llama2-7b-chat
model_name=llama3-8b-instruct
# model_name=llama3.1-8b-instruct
# llama2-7b-chat, llama3-8b-instruct, llama3.1-8b-instruct

topk=8
min_length=1k
max_length=32k
num_per=16

run_dir=output/niah
fname=mbr3
mkdir -p $run_dir/$model_name/$fname
pred_path=${run_dir}/${model_name}/${fname}/pred_chunk_num${chunk_num}_min${min_length}_max${max_length}_topk${topk}.json
echo "pred_path: ${pred_path}"

# greedy
if [ ! -e ${pred_path} ]; then
  python retrieve_and_inference_niah.py \
    hydra.run.dir=${run_dir}/${model_name}/${fname} \
    model_name=${model_name} \
    output_file=${pred_path} \
    task_name=${task_name} \
    topk=${topk} \
    method=${method} \
    max_length=${max_length} \
    min_length=${min_length} \
    num_per=${num_per} \
    generation_kwargs.num_return_sequences=1
fi 

# sample

# num_tests=5
# if [ ! -e ${pred_path} ]; then
#   python retrieve_and_inference_niah.py \
#     hydra.run.dir=${run_dir}/${model_name}/${fname} \
#     model_name=${model_name} \
#     output_file=${pred_path} \
#     task_name=${task_name} \
#     topk=${topk} \
#     method=${method} \
#     max_length=${max_length} \
#     min_length=${min_length} \
#     generation_kwargs.num_return_sequences=5 \
#     generation_kwargs.do_sample=True \
#     generation_kwargs.temperature=0.7
# fi 