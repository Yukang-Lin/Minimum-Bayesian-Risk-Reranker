#!/bin/bash
clear
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=$1
# task_names="passkey number_string kv_retrieval longbook_sum_eng longbook_choice_eng longbook_qa_eng longbook_qa_chn longdialogue_qa_eng math_find math_calc code_run code_debug"
# task_names="passkey number_string kv_retrieval longbook_choice_eng math_find code_debug"
# task_names="number_string kv_retrieval"
# task_names="longbook_choice_eng math_find code_debug"
task_names=$2
## infinitebench tasks
# retrieval tasks: passkey, number_string, kv_retrieval
# book tasks: longbook_sum_eng, longbook_choice_eng, longbook_qa_eng, longbook_qa_chn longdialogue_qa_eng
# math tasks: math_find, math_calc
# code tasks: code_run, code_debug
method=mbr3
# embed, mbr3, rerank_mbr3
# layer=12
topk=4
model_name=llama2-7b-chat
# model_name=llama3-8b-instruct
reverse=False
# model_name=llama3.1-8b-instruct
# llama2-7b-chat, llama3-8b-instruct, llama3.1-8b-instruct
# use_vllm=False
# gpu_memory_utilization=0.5

for task_name in $task_names; do
  echo "task_name: ${task_name}"
  run_dir=output/infinitebench/${task_name}
  # task_data_path=index_data/InfiniteBench/${task_name}/${task_name}.jsonl
  fname=mbr3
  mkdir -p $run_dir/$model_name/$fname
  if [ "$reverse" == "True" ]; then
    pred_path=${run_dir}/${model_name}/${fname}/pred_chunk_num${topk}_reversed.json
  else
    pred_path=${run_dir}/${model_name}/${fname}/pred_chunk_num${topk}.json
  fi
  echo "pred_path: ${pred_path}"

  if [ ! -e ${pred_path} ]; then
    python retrieve_and_inference_infinitebench.py \
      hydra.run.dir=${run_dir}/${model_name}/mbr3 \
      model_name=${model_name} \
      output_file=${pred_path} \
      task_name=${task_name} \
      topk=${topk} \
      method=${method} \
      reverse=${reverse}
  fi
done