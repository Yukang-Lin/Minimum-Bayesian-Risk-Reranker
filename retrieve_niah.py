
import hydra
import numpy as np
from tqdm import tqdm
from src.utils.misc import load_jsonl, save_jsonl, load_json, save_json, seed_everything
from src.tasks.niah.sf import build_niah_prompt
from src.tasks.niah.visualize import draw
from src.retrieve.mbr3 import RetrieverManager
from src.models.model_manager import ModelManager
from src.tasks.niah.config import NEEDLE, K, NIAH_FINAL_QUESTION, NIAH_DESCRIPTION
from loguru import logger
import os, time, json, re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# agent model: 
# opt-1.3b, gpt-neo-2.7B, Qwen2.5-0.5B

# import debugpy
# try:
#     print("Waiting for debugger attach")
#     debugpy.listen(("localhost", 9504))
#     debugpy.wait_for_client()
# except Exception as e:
#     print('error in debugpy')

class NIAHManager:
    def __init__(self, model_manager: ModelManager, retriever_manager: RetrieverManager, cfg) -> None:
        self.model_manager = model_manager
        self.retriever_manager = retriever_manager
        self.task_name = cfg.task_name
        self.output_file = cfg.output_file
        self.split_chunk = cfg.split_chunk
        self.chunk_num = cfg.chunk_num
        self.num_per = cfg.num_per
        _, self.tokenizer = model_manager.get_model_tokenizer()
        self.gen_length = len(self.tokenizer.encode(NEEDLE)) + 20
        sys_length = len(self.tokenizer.encode(NIAH_DESCRIPTION + NIAH_FINAL_QUESTION + NEEDLE))
        self.chunk_length = self.model_manager.n_tokens - sys_length - self.gen_length - cfg.buffer_length
        self.model_manager.generation_kwargs['max_new_tokens'] = self.gen_length
        self.sep = '\n'
        self.max_length = cfg.max_length
        self.min_length = cfg.min_length
        self.gap_length = cfg.gap_length
        self.window_split = cfg.window_split

        document_list = load_jsonl('index_data/needlebench/PaulGrahamEssays.jsonl', show_dir=False)
        context = '\n'.join([doc['text'] for doc in document_list])
        self.encoded_context = self.tokenizer.encode(context)

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
                entry = dict(context_length=length, depth_percent=depth)
                ## build context
                context = build_niah_prompt(encoded_context=self.encoded_context, context_length=length, depth=depth, tokenizer=self.tokenizer, sep=self.sep)
                ## chunking
                start = time.time()
                context_list = self.model_manager.split_long_context_to_chunk(text=context, token_nums=self.chunk_length//self.window_split, task_name=self.task_name)
                entry['chunk_time'] = time.time() - start
                # set label
                gt_idx = -1
                for i, context in enumerate(context_list):
                    if NEEDLE in context:
                        gt_idx = i
                        break
                assert gt_idx != -1, 'answer not in context'
                entry['gt_idx'] = gt_idx
                ## retrieve
                retrieve_func = self.retriever_manager.get_retrieve_func()
                start = time.time()
                ctxs = retrieve_func(query=query, context_list=context_list)
                entry['retrieve_time'] = time.time() - start
                entry['ctxs'] = ctxs
                try:
                    order = ctxs.index(gt_idx)
                except:
                    order = len(ctxs)
                entry['order'] = order
                entry['score'] = 1 - order / len(ctxs)
                self.dataset.append(entry)
                pbar.update(1)

        pbar.close()
        save_jsonl(self.output_file, self.dataset)

    def evaluate(self):
        scores = [item['score'] for item in self.dataset]
        avg_score = sum(scores) / len(scores)
        print('====== score statistic =====')
        print(f'= average score: {round(avg_score, 2)}')
        print('===========================')
        draw(dataset=self.dataset, output_path=self.output_file)

    def stat_time(self):
        data_len = len(self.dataset)
        chunk_time = [item.get('chunk_time', -1) for item in self.dataset]
        retrieve_time = [item.get('retrieve_time', -1) for item in self.dataset]
        
        chunk_time = sum(chunk_time) / data_len
        retrieve_time = sum(retrieve_time) / data_len

        print('====== time statistic =====')
        print(f'= chunk_time: {round(chunk_time, 2)}')
        print(f'= retrieve_time: {round(retrieve_time, 2)}')
        print('===========================')
        

@hydra.main(config_path="configs", config_name="retriever_and_inferencer_niah")
def main(cfg):
    logger.info(cfg)
    seed_everything(cfg.seed)
    model_manager = ModelManager(cfg)
    retriever_manager = RetrieverManager(model_manager, cfg)
    task_manager = NIAHManager(model_manager, retriever_manager, cfg)
    task_manager.predict()
    task_manager.evaluate()
    task_manager.stat_time()
    logger.info(f'finish task: {cfg.task_name}')

if __name__ == "__main__":
    main()
