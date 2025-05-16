import os
import json
import argparse
import numpy as np
from src.utils.misc import load_json, save_json
from src.tasks.infinitebench.eval_utils import DATA2MAXLEN, DATA2METRIC, get_answer

# import debugpy
# try:
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     raise e

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_names", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--run_dir_template", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default=None)
    return parser.parse_args(args)

def scorer(task_name, predictions, labels, model_name):
    metric = DATA2METRIC[task_name]
    eval_score = 0
    for label, pred in zip(labels, predictions):
        eval_score += metric(pred=pred, label=label, model_name=model_name)
    eval_score /= len(labels)
    return eval_score

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
        ground_truths = [get_answer(res, task_name) for res in results]

        score = scorer(task_name=task_name, predictions=predictions, labels=ground_truths, model_name=args.model_name) * 100
        print('='*20)
        print(f"Score on {task_name}: {score} %")
        scores[task_name] = round(score, 4)
        sum += score
    avg = sum / len(scores.keys()) * 100
    scores['average'] = avg

    if args.output_file:
        save_json(args.output_file, scores)
    