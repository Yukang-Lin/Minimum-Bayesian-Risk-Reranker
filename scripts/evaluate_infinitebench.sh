model_name=llama2-7b-chat
model_name=llama3-8b-instruct
method=mbr3
topk=4
# task_names="passkey,number_string,kv_retrieval,longbook_sum_eng,longbook_choice_eng,longbook_qa_eng,longbook_qa_chn,longdialogue_qa_eng,math_find,math_calc,code_run,code_debug"
task_names="passkey,number_string,kv_retrieval,longbook_choice_eng,longbook_qa_eng,longbook_qa_chn,longdialogue_qa_eng,math_find,math_calc,code_run,code_debug"
task_names="number_string"
# task_names="kv_retrieval"
# task_names='longbook_choice_eng'
# task_names='math_find'
# task_names='code_debug'
echo "eval results"
echo "============================="
# run_dir_template=output/longbench/{task_name}/${model_name}
run_dir_template=output/infinitebench/{task_name}/${model_name}/${method}
pred_file=pred_chunk_num${topk}.json
pred_file=pred_chunk_num${topk}_reversed.json
# output_file=output/longbench/eval_results_${model_name}.json
output_file=output/infinitebench/eval_results_${model_name}_${method}.json
python evaluate_results_infinitebench.py \
  --task_names ${task_names} \
  --model_name ${model_name} \
  --run_dir ${run_dir_template} \
  --pred_file ${pred_file} \
  --output_file ${output_file}

# python stat_time_longbench.py \
#   --task_names ${task_names} \
#   --run_dir ${run_dir_template} \
#   --prompt_file ${prompt_file} 