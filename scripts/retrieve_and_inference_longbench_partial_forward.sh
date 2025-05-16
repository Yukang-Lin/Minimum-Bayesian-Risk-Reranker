#!/bin/bash
clear
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=2
task_names='narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc'
task_names='repobench-p'
# task_names='trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p'
# task_names='passage_count passage_retrieval_en lcc repobench-p'
# task_names='qasper multifieldqa_en'
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
# model_name=llama2-7b-chat
model_name=llama3-8b-instruct
# model_name=llama3.1-8b-instruct
# llama2-7b-chat, llama3-8b-instruct, llama3.1-8b-instruct
topk=4
retrieve_topk=8
# topk=1
# retrieve_topk=3

self_define_lm=True
partial_retrieve=True
resume_forward=True
stop_at_layer=8
retrieve_mode=v1

for task_name in $task_names; do
  echo "task_name: ${task_name}"
  run_dir=output/longbench/${task_name}
  task_data_path=index_data/LongBench/${task_name}/test.jsonl
  fname=mbr3
  mkdir -p $run_dir/$model_name/$fname
  pred_path=${run_dir}/${model_name}/${fname}/pred_mode${retrieve_mode}_chunk_num${topk}_retrieve_topk${retrieve_topk}_layer${stop_at_layer}_1_reversed.json
  echo "pred_path: ${pred_path}"

  if [ ! -e ${pred_path} ]; then
    python retrieve_and_inference_longbench.py \
      hydra.run.dir=${run_dir}/${model_name}/mbr3 \
      model_name=${model_name} \
      output_file=${pred_path} \
      task_name=${task_name} \
      method=${method} \
      retrieve_mode=${retrieve_mode} \
      topk=${topk} \
      retrieve_topk=${retrieve_topk} \
      self_define_lm=${self_define_lm} \
      partial_retrieve=${partial_retrieve} \
      resume_forward=${resume_forward} \
      stop_at_layer=${stop_at_layer}
  fi
done