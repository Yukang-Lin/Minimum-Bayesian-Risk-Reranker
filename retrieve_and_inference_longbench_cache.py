import hydra
import hydra.utils as hu
from omegaconf import OmegaConf
import numpy as np
import torch
from tqdm import tqdm
from src.utils.misc import load_jsonl, save_jsonl, load_json, save_json, seed_everything
from src.tasks.longbench.configs.config import DATA2MAXLEN, DATA2PROMPT, DATA2PROMPT_CQ, DATA2SPLIT, DATA2METRIC
from src.retrieve.mbr3 import RetrieverManager
from src.models.model_manager import ModelManager
from loguru import logger
import os, time, json, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import debugpy
# try:
#     print("Waiting for debugger attach")
#     debugpy.listen(("localhost", 9504))
#     debugpy.wait_for_client()
# except Exception as e:
#     print('error in debugpy')
#     print(e)

class LongBenchManager:
    def __init__(self, model_manager: ModelManager, retriever_manager: RetrieverManager, cfg) -> None:
        self.model_manager = model_manager
        self.retriever_manager = retriever_manager
        self.task_name = cfg.task_name
        self.output_file = cfg.output_file
        self.dataset_path = cfg.dataset_path
        _, self.tokenizer = model_manager.get_model_tokenizer()
        self.gen_length = DATA2MAXLEN[self.task_name]
        self.sys_length = len(self.tokenizer.encode(DATA2PROMPT_CQ[self.task_name][0]+DATA2PROMPT_CQ[self.task_name][1]))
        self.buffer_length = cfg.buffer_length
        self.model_manager.generation_kwargs['max_new_tokens'] = self.gen_length
        self.parallel_num = cfg.parallel_num
        self.sep = '\n'
        # load dataset or chunks
        self.load_dataset()

    def load_dataset(self):
        self.dataset = load_jsonl(f'{self.dataset_path}/{self.task_name}/test.jsonl')

    def predict(self):
        template = DATA2PROMPT_CQ[self.task_name]
        if self.task_name in ['passage_count', 'passage_retrieval_en']:
            context_template, query_template, suffix = template
        else:
            context_template, query_template = template
        for entry in tqdm(self.dataset, total=len(self.dataset), desc='calculate query-loss and generate'):
            # get query
            query = entry['input']
            query = query_template.format(input=query) if self.task_name not in ['gov_report', 'multi_news', 'passage_count', 'lcc'] else query_template
            ## chunk
            context = entry.pop('context')
            query_tokens = len(self.tokenizer.encode(query, add_special_tokens=False))
            remaining_tokens = self.model_manager.n_tokens - self.gen_length - self.sys_length - self.buffer_length
            if query_tokens > remaining_tokens:
                query = self.model_manager.truncate_by_tokens(query, remaining_tokens)
                entry['chunk_num'] = 0
                entry['ctxs'] = []
                entry['chunk_time'] = 0
                context = '' # query is too long
                prompt = context_template.format(context=context) + query
                if self.task_name in ['passage_count', 'passage_retrieval_en']:
                    prompt = prompt + suffix
            else:
                token_nums = remaining_tokens - query_tokens
                token_nums //= self.retriever_manager.topk # split to small chunks
                start = time.time()
                context_list = self.model_manager.split_long_context_to_chunk(text=context, token_nums=token_nums, task_name=self.task_name)
                entry['chunk_time'] = time.time() - start
                entry['chunk_num'] = len(context_list)
                ## retrieve
                start = time.time()
                retrieve_func = self.retriever_manager.get_retrieve_func()
                context_list1 = [context_template.format(context=context) for context in context_list]
                ctxs = retrieve_func(query=query, context_list=context_list1)
                entry['retrieve_time'] = time.time() - start
                ctxs.reverse()
                entry['ctxs'] = ctxs
                retrieved_context_list = [context_list[i] for i in ctxs]
                ## inference
                if self.parallel_num == 1:
                    context = self.sep.join(retrieved_context_list)
                    if context[-1] == '\n':
                        context = context[:-1]
                    joined_context = context_template.format(context=context)
                    prompt = joined_context + query
                    if self.task_name in ['passage_count', 'passage_retrieval_en']:
                        prompt = prompt + suffix
                # add template
                # if self.task_name not in ["trec", "triviaqa", "samsum", "lcc", "repobench-p"]:
                #     prompt = add_template(self.model_manager.model_name, prompt)
            start = time.time()
            entry['generated'] = self.model_manager.do_generate(prompt)[0]
            entry['generate_time'] = time.time() - start
        
        save_json(self.output_file, self.dataset)

        # evaluate
        self.evaluate()
        self.stat_time()

    def evaluate(self):
        if self.dataset[0].get('generated', None) is None:
            raise Exception
        self.metric = DATA2METRIC[self.task_name]
        answers = [item['answers'] for item in self.dataset]
        predictions = [item['generated'] for item in self.dataset]
        all_classes = self.dataset[0]['all_classes']

        total_score = 0.
        for (prediction, ground_truths) in zip(predictions, answers):
            score = 0.
            prediction = prediction
            if self.task_name in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip('\n').split('\n')[0]
            for ground_truth in ground_truths:
                score = max(score, self.metric(prediction, ground_truth, all_classes=all_classes))
            total_score += score
        eval_score = round(100 * total_score / len(predictions), 2)
        print('====== evaluate =====')
        print('= ', eval_score)
        print('=====================')

    def stat_time(self):
        data_len = len(self.dataset)
        chunk_time = [item.get('chunk_time', -1) for item in self.dataset]
        retrieve_time = [item.get('retrieve_time', -1) for item in self.dataset]
        generate_time = [item.get('generate_time', -1) for item in self.dataset]
        
        chunk_time = sum(chunk_time) / data_len
        retrieve_time = sum(retrieve_time) / data_len
        generate_time = sum(generate_time) / data_len

        print('====== time statistic =====')
        print(f'= chunk_time: {round(chunk_time, 2)}')
        print(f'= retrieve_time: {round(retrieve_time, 2)}')
        print(f'= generate_time: {round(generate_time, 2)}')
        print('===========================')

@hydra.main(config_path="configs", config_name="retriever_and_inferencer_longbench")
def main(cfg):
    logger.info(cfg)
    seed_everything(cfg.seed)
    model_manager = ModelManager(cfg)
    retriever_manager = RetrieverManager(model_manager, cfg)
    task_manager = LongBenchManager(model_manager, retriever_manager, cfg)
    task_manager.predict()
    logger.info(f'finish task: {cfg.task_name}')

if __name__ == "__main__":
    main()
