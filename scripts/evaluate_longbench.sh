# model_name=llama2-7b-chat
model_name=llama3-8b-instruct
# model_name=llama3.1-8b-instruct
# method=rerank_mbr3_llm_hidden_state_layer20
# method=embed_bert_chunk_mean
# method=embed_bert_chunk_mean
method=mbr3
topk=4
retrieve_topk=8
# topk=1
# retrieve_topk=3
retrieve_mode=$1
layer=$2
# method=mbr3_chunk_loss
# task_names='narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p'
task_names='narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,trec,triviaqa,samsum,passage_count,passage_retrieval_en,lcc,repobench-p'
# task_names='narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news'
# task_names='passage_retrieval_en'
echo "eval results"
echo "============================="
# run_dir_template=output/longbench/{task_name}/${model_name}
run_dir_template=output/longbench/{task_name}/${model_name}/${method}
# pred_file=pred_chunk_num${topk}_retrieve_topk${retrieve_topk}_layer${layer}.json
# pred_file=pred_mode${retrieve_mode}_chunk_num${topk}_retrieve_topk${retrieve_topk}_layer${layer}_1.json
# pred_file=pred_chunk_num${topk}cache.json
# pred_file=pred_chunk_num${topk}_reversed.json
pred_file=pred_mode${retrieve_mode}_chunk_num${topk}_retrieve_topk${retrieve_topk}_layer${layer}_reversed.json
# output_file=output/longbench/eval_results_${model_name}.json
# output_file=output/longbench/eval_results_${model_name}_${method}.json
python evaluate_results_longbench.py \
  --task_names ${task_names} \
  --run_dir ${run_dir_template} \
  --pred_file ${pred_file}
  # --output_file ${output_file}

# python stat_time_longbench.py \
#   --task_names ${task_names} \
#   --run_dir ${run_dir_template} \
#   --prompt_file ${prompt_file} 