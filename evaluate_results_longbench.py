import os
import json
import argparse
import numpy as np
from src.utils.misc import load_json, save_json

# import debugpy
# try:
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     raise e

from src.tasks.longbench.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_names", type=str, required=True)
    parser.add_argument("--run_dir_template", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args(args)

def scorer(dataset, predictions, answers, all_classes):
    predictions = [prediction.lstrip('\n') for prediction in predictions]
    if dataset in ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news"]:
        predictions1 = [prediction.split('\n')[0] for prediction in predictions]
        predictions2 = [prediction.split('\n')[0].split('.')[0] for prediction in predictions]
    if dataset in ["trec", "triviaqa", "samsum"]:
        predictions = [prediction.split('\n')[0] for prediction in predictions]
            
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score

    if dataset in ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news"]:
        total_score1 = 0.
        for (prediction, ground_truths) in zip(predictions1, answers):
            score = 0.
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            total_score1 += score
        
        total_score2 = 0.
        for (prediction, ground_truths) in zip(predictions2, answers):
            score = 0.
            for ground_truth in ground_truths:
                score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
            total_score2 += score

        total_score = max(total_score, max(total_score1, total_score2))

    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    print("Evaluating on:", args.pred_file)
    task_names = [task.strip() for task in args.task_names.split(',')]
    sum = 0
    for task_name in task_names:
        run_dir = args.run_dir_template.format(task_name=task_name)
        load_path = f'{run_dir}/{args.pred_file}'
        if os.path.exists(load_path):
            results = load_json(load_path, show_dir=False)
        else:
            print(f"no Task {task_name} File! not found: {load_path}!")
            continue
        predictions, answers, lengths = [], [], []
        predictions = [res['generated'] for res in results]
        answers = [res['answers'] for res in results]
        all_classes = results[0]['all_classes']
        if results[0].get('length'):
            lengths = [res['length'] for res in results]
        score = scorer(task_name, predictions, answers, all_classes)
        print('='*20)
        print(f"Score on {task_name}: {score}")
        scores[task_name] = round(score, 2)
        sum += score
    avg = sum / len(scores.keys())
    scores['average'] = avg

    if args.output_file:
        save_json(args.output_file, scores)
    