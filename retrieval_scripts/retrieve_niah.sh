#!/bin/bash
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0

method=mbr3
# embed, mbr3, rerank_mbr3
# model_name=$1
# model_name=llama2-7b-chat
model_name=llama3-8b-instruct
# model_name=llama3.1-8b-instruct
# llama2-7b-chat, llama3-8b-instruct, llama3.1-8b-instruct

topk=4
retrieve_topk=8
window_split=16
min_length=16k
max_length=128k
gap_length=16k
num_per=16

self_define_lm=True
partial_retrieve=True
resume_forward=True
stop_at_layer=20
retrieve_mode=v4

run_dir=output/retrieval/niah
fname=mbr3
mkdir -p $run_dir/$model_name/$fname
pred_path=${run_dir}/${model_name}/${fname}/pred_mode_split16${retrieve_mode}_min${min_length}_max${max_length}_gap${gap_length}_k${topk}_rk${retrieve_topk}_l${stop_at_layer}.jsonl
# pred_path=${run_dir}/${model_name}/${fname}/retrieve_chunk_num${window_split}.jsonl
echo "pred_path: ${pred_path}"

python retrieve_niah.py \
  hydra.run.dir=${run_dir}/${model_name}/${fname} \
  model_name=${model_name} \
  output_file=${pred_path} \
  task_name=${task_name} \
  topk=${topk} \
  method=${method} \
  max_length=${max_length} \
  min_length=${min_length} \
  gap_length=${gap_length} \
  num_per=${num_per} \
  window_split=${window_split} \
  retrieve_mode=${retrieve_mode} \
  retrieve_topk=${retrieve_topk} \
  self_define_lm=${self_define_lm} \
  partial_retrieve=${partial_retrieve} \
  resume_forward=${resume_forward} \
  stop_at_layer=${stop_at_layer}