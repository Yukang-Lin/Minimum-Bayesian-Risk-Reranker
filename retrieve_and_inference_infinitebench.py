
import logging
# import faiss
import hydra
import hydra.utils as hu
from omegaconf import OmegaConf
import numpy as np
import torch
from tqdm import tqdm
import re
from functools import partial
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from src.utils.misc import load_jsonl, save_jsonl, load_json, save_json, seed_everything
from src.tasks.infinitebench.eval_utils import DATA2MAXLEN,  DATA2METRIC, DATA2PROMPT, get_answer, create_prompt
from src.retrieve.mbr3 import RetrieverManager
from src.models.model_manager import ModelManager
from loguru import logger
import os, time, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import debugpy
# try:
#     print("Waiting for debugger attach")
#     debugpy.listen(("localhost", 9504))
#     debugpy.wait_for_client()
# except Exception as e:
#     print('error in debugpy')
#     print(e)

class InfiniteBenchManager:
    def __init__(self, model_manager: ModelManager, retriever_manager: RetrieverManager, cfg) -> None:
        self.model_manager = model_manager
        self.retriever_manager = retriever_manager
        self.task_name = cfg.task_name
        self.output_file = cfg.output_file
        self.dataset_path = cfg.dataset_path
        _, self.tokenizer = model_manager.get_model_tokenizer()
        self.gen_length = DATA2MAXLEN[self.task_name]
        self.sys_length = len(self.tokenizer.encode(DATA2PROMPT[self.task_name]))
        self.buffer_length = cfg.buffer_length
        self.model_manager.generation_kwargs['max_new_tokens'] = self.gen_length
        self.sep = '\n'
        self.reverse = cfg.reverse
        # load dataset or chunks
        self.load_dataset()

    def load_dataset(self):
        window = str(self.model_manager.n_tokens//1024) + 'k'
        chunk_pth = f'{self.dataset_path}/{self.task_name}/chunk_{window}_{self.retriever_manager.topk}.jsonl'
        if os.path.exists(chunk_pth):
            self.dataset = load_jsonl(chunk_pth)
        else:
            self.dataset = load_jsonl(f'{self.dataset_path}/{self.task_name}/{self.task_name}.jsonl')

    def baseline_predict(self):
        # splitter = DATA2SPLIT.get(self.task_name, REGEX_EN)
        for entry in tqdm(self.dataset, total=len(self.dataset), desc='calculate query-loss and generate'):
            ## inference
            # entry: dict_keys(['id', 'context', 'input', 'answer', 'options'])
            start = time.time()
            # config prompt
            prompt = create_prompt(entry, self.task_name)
            prompt = self.model_manager.truncate_by_tokens(prompt, self.tokenizer, self.model_manager.n_tokens - self.gen_length - self.sys_length - self.buffer_length)
            entry['generated'] = self.model_manager.do_generate(prompt)[0]
            entry['generate_time'] = time.time() - start
            entry.pop('context', None)

        save_json(self.output_file, self.dataset)
        # evaluate
        self.evaluate()
        self.stat_time()

    def predict(self):
        # splitter = DATA2SPLIT.get(self.task_name, REGEX_EN)
        # self.dataset = self.dataset[17:]
        # logger.warn(f'use first 100 entries')
        for i, entry in enumerate(tqdm(self.dataset, total=len(self.dataset), desc='calculate query-loss and generate')):
            # get query
            # if self.task_name in ['passkey', 'number_string', 'kv_retrieval', 'math_find', 'code_debug']:
            #     query = entry['input']
            # elif self.task_name in ['longbook_choice_eng', 'longbook_qa_eng', 'longbook_qa_chn']:
            #     query = entry['question']
            # else:
            #     raise Exception(f'unsupported task: {self.task_name}')
            query = entry['input']
            ## STEP 1: chunk
            # context = entry.pop('context')
            query_length = len(self.tokenizer.encode(query, add_special_tokens=False))
            if query_length > self.model_manager.n_tokens - self.gen_length - self.sys_length -self.buffer_length:
                query = self.model_manager.truncate_by_tokens(query, self.model_manager.n_tokens - self.gen_length - self.sys_length - self.buffer_length)
                entry['ctxs'] = []
                entry['chunk_time'] = 0
                entry['context'] ='' # query is too long
            else:
                if entry.get('context_list', None) is not None:
                    context_list = entry.pop('context_list')
                # else:
                #     start = time.time()
                #     token_nums = self.model_manager.n_tokens - self.gen_length - self.sys_length - query_length - self.buffer_length
                #     token_nums //= self.retriever_manager.topk # split to small chunks
                #     context_list = self.model_manager.split_long_context_to_chunk(context, token_nums=token_nums, task_name=self.task_name)
                #     entry['chunk_time'] = time.time() - start
                ## STEP 2: retrieve
                start = time.time()
                retrieve_func = self.retriever_manager.get_retrieve_func()
                ctxs = retrieve_func(query=query, context_list=context_list)
                entry['retrieve_time'] = time.time() - start
                # reverse or not
                if self.reverse:
                    ctxs.reverse()
                entry['ctxs'] = ctxs
                retrieved_context_list = [context_list[i] for i in ctxs]
                context = self.sep.join(retrieved_context_list)
                context_len = len(self.tokenizer.encode(context))
                if context_len >= self.model_manager.n_tokens - self.gen_length - self.sys_length - self.buffer_length:
                    # normally this code will not be executed for preview length control
                    context = self.model_manager.truncate_by_tokens(context, self.model_manager.n_tokens - self.gen_length - self.sys_length - self.buffer_length)
                    logger.warning(f'idx: {i}, context is too long: {context_len} > {self.model_manager.n_tokens - self.gen_length - self.sys_length - self.buffer_length}')
                entry['context'] = context
            ## STEP 3: inference
            # entry: dict_keys(['id', 'context', 'input', 'answer', 'options'])
            start = time.time()
            # config prompt
            prompt = create_prompt(entry, self.task_name)
            entry['generated'] = self.model_manager.do_generate(prompt)[0]
            entry['generate_time'] = time.time() - start
            entry.pop('context', None)

        save_json(self.output_file, self.dataset)

        # evaluate
        self.evaluate()
        self.stat_time()

    def evaluate(self): 
        if self.dataset[0].get('generated', None) is None:
            raise Exception
        self.metric = DATA2METRIC[self.task_name]
        labels = [get_answer(item, self.task_name) for item in self.dataset]
        predictions = [item['generated'] for item in self.dataset]
        
        eval_score = 0
        for label, pred in zip(labels, predictions):
            eval_score += self.metric(pred=pred, label=label, model_name=self.model_manager.model_name)
        eval_score /= len(labels)
        print('====== evaluate =====')
        print('= ', eval_score)
        print('=====================')

    def stat_time(self):
        data_len = len(self.dataset)
        # chunk_time = [item.get('chunk_time', -1) for item in self.dataset]
        # retrieve_time = [item.get('retrieve_time', -1) for item in self.dataset]
        generate_time = [item.get('generate_time', -1) for item in self.dataset]
        
        # chunk_time = sum(chunk_time) / data_len
        # retrieve_time = sum(retrieve_time) / data_len
        generate_time = sum(generate_time) / data_len

        print('====== time statistic =====')
        # print(f'= chunk_time: {round(chunk_time, 2)}')
        # print(f'= retrieve_time: {round(retrieve_time, 2)}')
        print(f'= generate_time: {round(generate_time, 2)}')
        print('===========================')

@hydra.main(config_path="configs", config_name="retriever_and_inferencer_infinitebench")
def main(cfg):
    logger.info(cfg)
    seed_everything(cfg.seed)
    model_manager = ModelManager(cfg)
    retriever_manager = RetrieverManager(model_manager, cfg)
    task_manager = InfiniteBenchManager(model_manager, retriever_manager, cfg)
    # task_manager.baseline_predict()
    task_manager.predict()
    logger.info(f'finish task: {cfg.task_name}')

if __name__ == "__main__":
    main()
