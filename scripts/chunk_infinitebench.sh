#!/bin/bash
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2
task_names="passkey number_string kv_retrieval longbook_sum_eng longbook_choice_eng longbook_qa_eng longbook_qa_chn longdialogue_qa_eng math_find math_calc code_run code_debug"
task_names="passkey number_string kv_retrieval longbook_choice_eng math_find code_debug"
task_names=$1
# task_names="number_string kv_retrieval"
# task_names="longbook_choice_eng math_find code_debug"
## infinitebench tasks
# retrieval tasks: passkey, number_string, kv_retrieval
# book tasks: longbook_sum_eng, longbook_choice_eng, longbook_qa_eng, longbook_qa_chn longdialogue_qa_eng
# math tasks: math_find, math_calc
# code tasks: code_run, code_debug
method=mbr3
# embed, mbr3, rerank_mbr3
# layer=12
topk=1
model_name=llama2-7b-chat
# model_name=llama3-8b-instruct
# model_name=llama3.1-8b-instruct
# llama2-7b-chat, llama3-8b-instruct, llama3.1-8b-instruct
# use_vllm=False
# gpu_memory_utilization=0.5

for task_name in $task_names; do
  echo "task_name: ${task_name}"
  task_data_path=index_data/InfiniteBench/${task_name}/test.jsonl
  fname=mbr3
  chunk_path=index_data/InfiniteBench/${task_name}/chunk_8k_${topk}.jsonl

  if [ ! -e ${chunk_path} ]; then 
    python chunk_infinitebench.py \
      model_name=${model_name} \
      output_file=${chunk_path} \
      task_name=${task_name} \
      topk=${topk} \
      method=${method} 
  fi
done