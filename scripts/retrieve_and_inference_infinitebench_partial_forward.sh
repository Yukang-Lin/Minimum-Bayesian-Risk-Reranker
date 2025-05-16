#!/bin/bash
clear
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$1
task_names=$2
## infinitebench tasks
# retrieval tasks: passkey, number_string, kv_retrieval
# book tasks: longbook_sum_eng, longbook_choice_eng, longbook_qa_eng, longbook_qa_chn longdialogue_qa_eng
# math tasks: math_find, math_calc
# code tasks: code_run, code_debug
method=mbr3
model_name=llama2-7b-chat
model_name=llama3-8b-instruct
# model_name=llama3.1-8b-instruct
# llama2-7b-chat, llama3-8b-instruct, llama3.1-8b-instruct
# use_vllm=False
# gpu_memory_utilization=0.5
window_split=2
# topk=4
# retrieve_topk=8
topk=1
retrieve_topk=4

self_define_lm=True
partial_retrieve=True
resume_forward=True
stop_at_layer=$3
retrieve_mode=$4

reverse=True

for task_name in $task_names; do
  echo "task_name: ${task_name}"
  run_dir=output/infinitebench/${task_name}
  # task_data_path=index_data/InfiniteBench/${task_name}/${task_name}.jsonl
  fname=mbr3
  mkdir -p $run_dir/$model_name/$fname
  if [ "$reverse" == "True" ]; then
    pred_path=${run_dir}/${model_name}/${fname}/pred_mode${retrieve_mode}_window_split${window_split}_topk${topk}_retrieve_topk${retrieve_topk}_layer${stop_at_layer}_reversed.json
  else
    pred_path=${run_dir}/${model_name}/${fname}/pred_mode${retrieve_mode}_window_split${window_split}_topk${topk}_retrieve_topk${retrieve_topk}_layer${stop_at_layer}.json
  fi
  echo "pred_path: ${pred_path}"

  if [ ! -e ${pred_path} ]; then
    python retrieve_and_inference_infinitebench.py \
      hydra.run.dir=${run_dir}/${model_name}/mbr3 \
      model_name=${model_name} \
      output_file=${pred_path} \
      task_name=${task_name} \
      method=${method} \
      retrieve_mode=${retrieve_mode} \
      window_split=${window_split} \
      topk=${topk} \
      retrieve_topk=${retrieve_topk} \
      self_define_lm=${self_define_lm} \
      partial_retrieve=${partial_retrieve} \
      resume_forward=${resume_forward} \
      stop_at_layer=${stop_at_layer} \
      reverse=${reverse}
  fi
done