#!/bin/bash
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
task_names='narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p'
# task_names='gov_report multi_news passage_count lcc repobench-p'
# task_names='passage_count'
# task_names='narrativeqa hotpotqa 2wikimqa musique trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p'
# task_names='repobench-p'
# task_names='triviaqa samsum passage_count passage_retrieval_en lcc repobench-p'
## longbench: only test on en tasks
# single-document qa: narrativeqa, qasper, multifieldqa_en
# multi-document qa: hotpotqa, 2wikimqa, musique
# summarization: gov_report, qmsum, multi_news
# few-shot learning: trec, triviaqa, samsum
# synthetic tasks: passage_count, passage_retrieval_en
# coding: lcc, repobench-p

method=mbr3
# embed, mbr3, rerank_mbr3
# layer=16
topk=1
model_name=llama2-7b-chat
# model_name=llama3-8b-instruct
# model_name=llama3.1-8b-instruct
# llama2-7b-chat, llama3-8b-instruct, llama3.1-8b-instruct

for task_name in $task_names; do
  echo "task_name: ${task_name}"
  run_dir=output/longbench/${task_name}
  task_data_path=index_data/LongBench/${task_name}/test.jsonl
  fname=mbr3
  mkdir -p $run_dir/$model_name/$fname
  pred_path=${run_dir}/${model_name}/${fname}/pred_chunk_num${topk}cache.json
  echo "pred_path: ${pred_path}"

  # if [ -e ${pred_path} ]; then
  #   rm ${pred_path}
  # fi
  # python retrieve_and_inference_longbench_cache.py \
  #   hydra.run.dir=${run_dir}/${model_name}/mbr3 \
  #   model_name=${model_name} \
  #   output_file=${pred_path} \
  #   task_name=${task_name} \
  #   topk=${topk} \
  #   method=${method}

  if [ ! -e ${pred_path} ]; then
    python retrieve_and_inference_longbench_cache.py \
      hydra.run.dir=${run_dir}/${model_name}/mbr3 \
      model_name=${model_name} \
      output_file=${pred_path} \
      task_name=${task_name} \
      topk=${topk} \
      method=${method}
  fi
done