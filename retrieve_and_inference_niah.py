
import logging
# import faiss
import hydra
import hydra.utils as hu
from omegaconf import OmegaConf
import numpy as np
import torch
from tqdm import tqdm
import re
from src.utils.misc import load_jsonl, save_jsonl, load_json, save_json, seed_everything
from src.tasks.niah.visualize import draw
from src.tasks.niah.sf import build_niah_prompt
from src.retrieve.mbr3 import RetrieverManager
from src.models.model_manager import ModelManager
from src.tasks.niah.config import NEEDLE, NIAH_KEY_WORDS, K, NIAH_FINAL_QUESTION, NIAH_DESCRIPTION, REGEX_EN
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

class NIAHManager:
    def __init__(self, model_manager: ModelManager, retriever_manager: RetrieverManager, cfg) -> None:
        self.task_name = cfg.task_name
        self.model_manager = model_manager
        self.retriever_manager = retriever_manager
        self.task_name = cfg.task_name
        self.output_file = cfg.output_file
        self.split_chunk = cfg.split_chunk
        self.chunk_num = cfg.chunk_num
        self.num_per = cfg.num_per
        # self.dataset_path = cfg.dataset_path
        # self.load_dataset()
        _, self.tokenizer = model_manager.get_model_tokenizer()
        self.gen_length = len(self.tokenizer.encode(NEEDLE)) + 20
        sys_length = len(self.tokenizer.encode(NIAH_DESCRIPTION + NIAH_FINAL_QUESTION + NEEDLE))
        self.chunk_length = self.model_manager.n_tokens - sys_length - self.gen_length - cfg.buffer_length
        self.model_manager.generation_kwargs['max_new_tokens'] = self.gen_length
        self.sep = '\n'
        self.max_length = cfg.max_length
        self.min_length = cfg.min_length
        self.gap_length = cfg.gap_length

        document_list = load_jsonl('index_data/needlebench/PaulGrahamEssays.jsonl', show_dir=False)
        context = '\n'.join([doc['text'] for doc in document_list])
        self.encoded_context = self.tokenizer.encode(context)
    # def load_dataset(self):
    #     self.dataset = load_jsonl(f'{self.dataset_path}/{self.task_name}/test.jsonl')

    def origin_predict(self):
        max_length = int(self.max_length.replace("k", '')) * K
        min_length = int(self.min_length.replace("k", '')) * K
        assert min_length <= max_length, f'min_length: {min_length}, max_length: {max_length}'
        gap = int(self.gap_length.replace("k", '')) * K
        num_per = self.num_per
        depth_percent = 1 / num_per
        length_list = [i for i in range(min_length, max_length + 1, gap)]

        pbar = tqdm(length_list*num_per)
        self.dataset = []
        for length in length_list:
            depths = [depth_percent * i for i in range(1, num_per + 1)]
            for depth in depths:
                entry = dict()
                ## build context
                prompt = build_niah_prompt(encoded_context=self.encoded_context, context_length=length, depth=depth, tokenizer=self.tokenizer, sep=self.sep)
                prompt = NIAH_DESCRIPTION + self.sep + prompt + self.sep + NIAH_FINAL_QUESTION
                start = time.time()
                responses = self.model_manager.do_generate(prompt)
                entry['generate_time'] = time.time() - start
                entry['context_length'] = length
                entry['depth_percent'] = depth
                entry['prompt_len'] = len(self.tokenizer.encode(prompt))
                entry['generated'] = responses
                
                self.dataset.append(entry)
                pbar.update(1)

        pbar.close()
        save_json(self.output_file, self.dataset)
        logger.info('do evaluation')
        print('====== retrieve counter ======')
        self.evaluate()
        self.stat_time()

    def predict(self):
        max_length = int(self.max_length.replace("k", '')) * K
        min_length = int(self.min_length.replace("k", '')) * K
        assert min_length <= max_length, f'min_length: {min_length}, max_length: {max_length}'
        gap = int(self.gap_length.replace("k", '')) * K
        num_per = self.num_per
        depth_percent = 1 / num_per
        length_list = [i for i in range(min_length, max_length + 1, gap)]

        query = NIAH_FINAL_QUESTION

        pbar = tqdm(length_list*num_per)
        self.dataset = []
        retrieve_dict = {}
        for length in length_list:
            retrieve_dict[length] = {}
            depths = [depth_percent * i for i in range(1, num_per + 1)]
            for depth in depths:
                entry = dict()
                ## build context
                context = build_niah_prompt(context_length=length, depth=depth, tokenizer=self.tokenizer, sep=self.sep)
                ## chunking
                start = time.time()
                context_list, idx = self.model_manager.split_long_context_to_chunk(text=context, token_nums=self.chunk_length//self.retriever_manager.topk, find_str=NEEDLE, task_name=self.task_name)
                # split to smaller chunk
                # if self.split_chunk:
                #     new_context_list = []
                #     for id, context in enumerate(context_list):
                #         chunks, idx1 = self.model_manager.split_chunk(text=context, chunk_num=self.chunk_num, task_name=self.task_name, find_str=NEEDLE)
                #         if id == idx:
                #             assert idx1 != -1, "cannot find needle in the context"
                #             new_idx = len(new_context_list) + idx1
                #         new_context_list.extend(chunks)
                #     idx = new_idx
                #     context_list = new_context_list

                entry['chunk_time'] = time.time() - start
                ## retrieve
                start = time.time()
                retrieve_func = self.retriever_manager.get_retrieve_func()
                ctxs = retrieve_func(query=query, context_list=context_list)
                entry['ctxs'] = ctxs
                if idx in ctxs:
                    retrieve_dict[length][depth] = 1
                else:
                    retrieve_dict[length][depth] = 0
                retrieved_context_list = [context_list[i] for i in ctxs]
                entry['retrieve_time'] = time.time() - start
                ## inference
                start = time.time()
                # add template
                prompt = self.sep.join(retrieved_context_list)
                prompt = NIAH_DESCRIPTION + self.sep + prompt + self.sep + NIAH_FINAL_QUESTION
                # messages = [
                #     {"role": "system", "content": NIAH_DESCRIPTION},
                #     {"role": "user", "content": prompt},
                #     {"role": "user", "content": NIAH_FINAL_QUESTION}
                # ]
                # prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # prompt = add_template(self.model_manager.model_name, prompt)
                responses = self.model_manager.do_generate(prompt)
                entry['context_length'] = length
                entry['depth_percent'] = depth
                entry['prompt_len'] = len(self.tokenizer.encode(prompt))
                entry['generated'] = responses
                entry['generate_time'] = time.time() - start
                
                self.dataset.append(entry)
                pbar.update(1)

        pbar.close()
        logger.info('do evaluation')
        print('====== retrieve counter ======')
        print('= ', retrieve_dict)
        # print('= ', retrieve_counter / (len(self.dataset) * len(responses)) * 100, ' %')
        print('=============================')
        save_json('retrieve_dict.json', retrieve_dict)
        self.evaluate()
        self.stat_time()

    def evaluate(self):
        tmp = self.dataset[0].get('generated', None)
        if tmp is None:
            raise Exception
        response_num = len(tmp)
        avg_score = 0
        for item in self.dataset:
            correctness = [all([keyword.lower() in r.lower() for keyword in NIAH_KEY_WORDS]) for r in item['generated']]
            score = sum(correctness) / response_num
            item['score'] = score
            avg_score += score
        avg_score /= (response_num * len(self.dataset))
        print('====== avg_score ======')
        print('= ', avg_score)
        print('=======================')

        save_json(self.output_file, self.dataset)
        draw(dataset=self.dataset, output_path = self.output_file)

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
        

@hydra.main(config_path="configs", config_name="retriever_and_inferencer_niah")
def main(cfg):
    logger.info(cfg)
    seed_everything(cfg.seed)
    model_manager = ModelManager(cfg)
    retriever_manager = RetrieverManager(model_manager, cfg)
    task_manager = NIAHManager(model_manager, retriever_manager, cfg)
    task_manager.origin_predict()
    # task_manager.predict()
    logger.info(f'finish task: {cfg.task_name}')

if __name__ == "__main__":
    main()
