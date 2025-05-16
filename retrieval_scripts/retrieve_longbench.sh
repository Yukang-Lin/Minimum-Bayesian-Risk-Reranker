#!/bin/bash
clear
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2
task_name='passage_retrieval_en'

method=mbr3
# embed, mbr3, rerank_mbr3
# layer=16
model_name=llama2-7b-chat
# model_name=llama3-8b-instruct
# model_name=llama3.1-8b-instruct
# llama2-7b-chat, llama3-8b-instruct, llama3.1-8b-instruct

topk=4
retrieve_topk=8
self_define_lm=True
partial_retrieve=True
resume_forward=True
stop_at_layer=16
retrieve_mode=v3

run_dir=output/retrieval/${task_name}
fname=mbr3
mkdir -p $run_dir/$model_name/$fname
# pred_path=${run_dir}/${model_name}/${fname}/pred_chunk_num${topk}.json
pred_path=${run_dir}/${model_name}/${fname}/pred_mode_${retrieve_mode}_k${topk}_rk${retrieve_topk}_l${stop_at_layer}.jsonl

echo "pred_path: ${pred_path}"

if [ ! -e ${pred_path} ]; then
  python retrieve_longbench.py \
    hydra.run.dir=${run_dir}/${model_name}/mbr3 \
    model_name=${model_name} \
    output_file=${pred_path} \
    task_name=${task_name} \
    topk=${topk} \
    method=${method}
fi