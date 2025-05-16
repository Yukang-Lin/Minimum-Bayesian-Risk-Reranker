
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
        # load dataset or chunks
        self.load_dataset()

    def load_dataset(self):
        self.dataset = load_jsonl(f'{self.dataset_path}/{self.task_name}/{self.task_name}.jsonl')

    def chunk(self):
        # splitter = DATA2SPLIT.get(self.task_name, REGEX_EN)
        sys_prompt = DATA2PROMPT[self.task_name]
        for entry in tqdm(self.dataset, total=len(self.dataset), desc='chunking'):
            # get query
            # if self.task_name in ['passkey', 'number_string', 'kv_retrieval', 'math_find', 'code_debug']:
            #     query = entry['input']
            # elif self.task_name in ['longbook_choice_eng', 'longbook_qa_eng', 'longbook_qa_chn']:
            #     query = entry['question']
            # else:
            #     raise Exception(f'unsupported task: {self.task_name}')
            query = entry['input']
            ## step 1: chunk
            context = entry.pop('context')
            query_length = len(self.tokenizer.encode(query, add_special_tokens=False))
            if query_length > self.model_manager.n_tokens - self.gen_length - self.sys_length - self.buffer_length:
                query = self.model_manager.truncate_by_tokens(query, self.model_manager.n_tokens - self.gen_length - self.sys_length - self.buffer_length)
                entry['ctxs'] = []
                entry['chunk_time'] = 0
                entry['context'] ='' # query is too long
            else:
                start = time.time()
                # FORMATTING
                sys_length=len(self.tokenizer.encode(sys_prompt.format(
                    context='{context}',
                    question='{question}',
                    OPTION_A=entry["options"][0],
                    OPTION_B=entry["options"][1],
                    OPTION_C=entry["options"][2],
                    OPTION_D=entry["options"][3],
                )))
                token_nums = self.model_manager.n_tokens - self.gen_length - sys_length - query_length - self.buffer_length
                token_nums //= self.retriever_manager.topk # split to small chunks
                context_list = self.model_manager.split_long_context_to_chunk(context, token_nums=token_nums, task_name=self.task_name)
                entry['chunk_time'] = time.time() - start
            entry.pop('context', None)
            entry['context_list'] = context_list
        save_jsonl(self.output_file, self.dataset)

@hydra.main(config_path="configs", config_name="retriever_and_inferencer_infinitebench")
def main(cfg):
    logger.info(cfg)
    seed_everything(cfg.seed)
    model_manager = ModelManager(cfg)
    retriever_manager = RetrieverManager(model_manager, cfg)
    task_manager = InfiniteBenchManager(model_manager, retriever_manager, cfg)
    # task_manager.baseline_predict()
    task_manager.chunk()
if __name__ == "__main__":
    main()
