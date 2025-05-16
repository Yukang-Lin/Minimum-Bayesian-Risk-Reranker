
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
from src.tasks.longbench.configs.config import DATA2MAXLEN, DATA2PROMPT, DATA2SPLIT, DATA2METRIC
from src.retrieve.mbr3 import RetrieverManager
from src.models.model_manager import ModelManager
from loguru import logger
import os, time, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# import debugpy
# try:
#     print("Waiting for debugger attach")
#     debugpy.listen(("localhost", 9503))
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
        self.sys_length = len(self.tokenizer.encode(DATA2PROMPT[self.task_name]))
        self.buffer_length = cfg.buffer_length
        self.sep = '\n'
        # load dataset or chunks
        self.load_dataset()

    def load_dataset(self):
        self.dataset = load_jsonl(f'{self.dataset_path}/{self.task_name}/test.jsonl')

    def predict(self):
        for entry in tqdm(self.dataset, total=len(self.dataset), desc='chunk and calculate query-loss'):
            # get query
            query = entry['input']
            query_length = len(self.tokenizer.encode(query, add_special_tokens=False))
            answer = entry['answers'][0]
            ## chunk
            context = entry.pop('context')
            # token_nums = self.model_manager.n_tokens - self.gen_length - self.sys_length - query_length - self.buffer_length
            # token_nums //= self.retriever_manager.topk # split to small chunks
            # context_list = self.model_manager.split_long_context_to_chunk(text=context, token_nums=token_nums, task_name=self.task_name)
            splitter = 'Paragraph'
            context_list = context.split(splitter)[1:]
            context_list = [splitter + sent for sent in context_list]
            entry['context_list'] = context_list
            gt_idx = -1
            for i, context in enumerate(context_list):
                if answer in context:
                    gt_idx = i
                    break
            assert gt_idx != -1, 'answer not in context'
            ## retrieve
            retrieve_func = self.retriever_manager.get_retrieve_func()
            start = time.time()
            ctxs = retrieve_func(query=query, context_list=context_list)
            entry['retrieve_time'] = time.time() - start
            entry['gt_idx'] = gt_idx
            entry['ctxs'] = ctxs
            order = ctxs.index(gt_idx)
            entry['order'] = order

        save_json(self.output_file, self.dataset)
        self.stat_time()

    def stat_time(self):
        data_len = len(self.dataset)
        retrieve_time = [item.get('retrieve_time', -1) for item in self.dataset]
        
        retrieve_time = sum(retrieve_time) / data_len

        print('====== time statistic =====')
        print(f'= retrieve_time: {round(retrieve_time, 2)}')
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
