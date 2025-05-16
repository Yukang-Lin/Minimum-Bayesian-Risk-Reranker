from typing import List
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from functools import partial
import time
from tqdm import tqdm
from loguru import logger
from src.models.modeling_llama_mbr3 import BaseModelOutputWithPastForMBR3
import heapq
import random

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# LEGACY
def topk_chunk_in_token_level(scores: List[torch.FloatTensor], topk: int):
    """
    retrieve topk chunks in token level scores
    return: topk chunks index
    """
    if len(scores) <= topk:
        return list(range(len(scores)))
    # Get the maximum token score for each chunk
    max_scores = [torch.max(score.float()) for score in scores]
    # Combine max scores into a tensor
    max_scores_tensor = torch.tensor(max_scores)
    # print(max_scores_tensor.shape)
    
    # Get the indices of the topk chunks with highest max scores
    _, topk_indices = torch.topk(max_scores_tensor, k=topk)
    
    return topk_indices.tolist()

class CacheManager:
    def __init__(self, k):
        self.k = k
        self.heap = []  # Min-heap storing scores

    def push(self, score, indice, chunk_len, cache):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (score, indice, chunk_len, cache))
        else:
            # Replace the lowest score
            popped_item = heapq.heappushpop(self.heap, (score, indice, chunk_len, cache))
            self._clear_cache(popped_item)

    def _clear_cache(self, item):
        del item
        torch.cuda.empty_cache()

    def get_topk(self):
        return self.heap

    def clear(self):
        """
        Clear all cached elements and reset the manager state.
        Releases all cached resources.
        """
        # Release all cache resources
        for item in self.heap:
            self._clear_cache(item)
        # Reset all state
        self.heap = []

    # def _release_cache(self, cache):
    #     if torch.is_tensor(cache):
    #         cache.cpu()
    #     del cache
    #     torch.cuda.empty_cache()

class RetrieverManager:
    def __init__(self, model_manager, cfg):
        
        self.model_manager = model_manager
        self.cache_manager = CacheManager(cfg.retrieve_topk)
        _, self.tokenizer = model_manager.get_model_tokenizer()
        self.topk = cfg.topk
        self.partial_retrieve = cfg.partial_retrieve
        self.retrieve_mode = cfg.retrieve_mode
        self.return_dict = True
        self.output_attentions = False
        self.output_last_attention = True if self.retrieve_mode == 'v3' else False
        self.output_hidden_states = False
        # split retrieve and inference
        self.use_cache = True if self.retrieve_mode == 'v1' or self.retrieve_mode == 'v2' else False
        self.resume_forward = cfg.resume_forward
            
        self.retrieve_topk = cfg.retrieve_topk
        self.stop_at_layer = cfg.stop_at_layer
        self.use_vllm = cfg.use_vllm
        self.sep = '\n'
        self.method = cfg.method
        logger.info(f'Using {self.method} to retrieve')

    def get_retrieve_func(self):
        if self.partial_retrieve:
            if self.resume_forward:
                retrieve_func = self.retrieve_by_query_loss_with_partial_forward_and_resume
            else:
                retrieve_func = self.retrieve_by_query_loss_with_partial_forward
        else:
            retrieve_func = self.retrieve_by_query_loss
        return retrieve_func

    def middle_layer_similarity(self, chunk_len: int, past_key_value: Optional[List[dict[str, torch.FloatTensor]]], last_attention: torch.FloatTensor, query_ids: torch.Tensor, logits: torch.FloatTensor):
        """
        calculate similarity between query and kvPair
        selecting metrics: k*k^T, v*v^T, attention_score
        return: similarity score
        """
        # use key as default
        mode = self.retrieve_mode
        if mode == 'v1':
            key_tensor = past_key_value[-1][0]
            # split away query and context
            chunk_tensor = key_tensor[0, :, :chunk_len, :]
            query_tensor = key_tensor[0, :, chunk_len:, :]
            # calculate similarity
            sim_scores = torch.matmul(chunk_tensor, query_tensor.transpose(-1, -2))
            # pooling over all heads
            sim_scores = torch.mean(sim_scores, dim=0)
            # pooling over all query tokens
            sim_scores = torch.mean(sim_scores, dim=-1)
            sim_scores = torch.max(sim_scores, dim=-1)[0].item()
        elif mode == 'v2':
            # use value as default
            value_tensor = past_key_value[-1][1]
            # split away query and context
            chunk_tensor = value_tensor[0, :, :chunk_len, :]
            query_tensor = value_tensor[0, :, -chunk_len:, :]
            # calculate similarity
            sim_scores = torch.matmul(chunk_tensor, query_tensor.transpose(-1, -2))
            # pooling over all heads
            sim_scores = torch.mean(sim_scores, dim=0)
            # pooling over all query tokens
            sim_scores = torch.mean(sim_scores, dim=-1)
            sim_scores = torch.max(sim_scores, dim=-1)[0].item()
        elif mode == 'v3':
            # split away query and context
            chunk_query_tensor = last_attention[0, :, chunk_len:, :chunk_len]
            # pooling over all heads
            sim_scores = torch.mean(chunk_query_tensor, dim=0)
            # pooling over all query tokens
            sim_scores = torch.mean(sim_scores, dim=0)
            sim_scores = torch.max(sim_scores).item()
        elif mode == 'v4':
            sim_scores = self.model_manager.get_query_loss(encoded_query=query_ids, logits=logits[0, chunk_len - 1: -1, :])
            # align with other metrics
            sim_scores *= -1
        return sim_scores

    # ABORT
    # def retrieve_by_query_loss(self, query: str, context_list: List[str], partial_retrieve=False, resume_forward=True):
    #     if partial_retrieve:
    #         if resume_forward:
    #             return self.retrieve_by_query_loss_with_partial_forward_and_resume(query=query, context_list=context_list, sep=self.sep)
    #         else:
    #             return self.retrieve_by_query_loss_with_partial_forward(query=query, context_list=context_list, sep=self.sep)
    #     else:
    #         loss_list = self.model_manager.get_context_query_loss(query, context_list, sep=self.sep)
            
    #     global_indices = np.argsort(np.array(loss_list))[:self.topk]
    #     return global_indices.tolist()

    def retrieve_by_query_loss(self, query: str, context_list: List[str], sep: str = '\n'):
        """
        retrieve by query loss
        """
        unique_context_list = list(set(context_list))
        unique_loss_list = self.model_manager.get_context_query_loss(query, unique_context_list, sep=sep)
        context_loss_dict = dict(zip(unique_context_list, unique_loss_list))
        loss_list = [context_loss_dict[context] for context in context_list]
        global_indices = np.argsort(np.array(loss_list))[:self.topk]
        return global_indices.tolist()

    def retrieve_by_query_loss_with_partial_forward_and_resume(self, query: str, context_list: List[str], sep: str = '\n'):
        """
        step 1: partial forward to get key-value pairs in lower layer
        step 2: lower-layer retrieval with key-value pairs
        step 3: resume forward to the rest layers to get the final logits
        step 4: re-rank with the retrieved chunk
        """
        # TODO: change to batch version
        # deduplicated
        unique_context_list = list(set(context_list))
        # get key-value pairs in middle layer
        for indice, context in enumerate(unique_context_list):
            # input text should be context + sep + query
            text = context + sep + query
            chunk_len = len(self.tokenizer.encode(context+sep))
            encoded_text = self.tokenizer(text, return_tensors='pt')
            input_ids = encoded_text.input_ids
            query_ids = input_ids[0, chunk_len:]
            partial_forward_cache = self.model_manager.do_partial_forward(input_ids=input_ids, stop_at_layer=self.stop_at_layer, output_last_attention=self.output_last_attention, \
                                    output_attentions=self.output_attentions, output_hidden_states=self.output_hidden_states, \
                                    use_cache=self.use_cache, return_dict=self.return_dict)
            
            last_attention = partial_forward_cache.last_attention
            past_key_value = partial_forward_cache.past_key_values
            logits = partial_forward_cache.logits
            score = self.middle_layer_similarity(chunk_len=chunk_len, past_key_value=past_key_value, last_attention=last_attention, query_ids=query_ids, logits=logits)
            self.cache_manager.push(score, indice, chunk_len, partial_forward_cache)
        # get topk indices
        retrieved_topk_tuple = self.cache_manager.get_topk()
        retrieved_loss_list = [t[0] for t in retrieved_topk_tuple]
        retrieved_indice_list = [t[1] for t in retrieved_topk_tuple]
        retrieved_chunk_len_list = [t[2] for t in retrieved_topk_tuple]
        retrieved_partial_forward_cache_list = [t[3] for t in retrieved_topk_tuple]
        self.cache_manager.clear()
        
        querylossList = []
        retrieved_indice_context_dict = {indice: unique_context_list[indice] for indice in retrieved_indice_list}
        # context_loss_dict = dict(zip(context_list, unique_loss_list))
        # loss_list = [context_loss_dict[context] for context in context_list]
        # global_indices = np.argsort(np.array(loss_list))
        retrieve_map = dict(zip(range(len(retrieved_indice_list)), retrieved_indice_list))
        # continue forward to the rest layers
        for i, partial_forward_cache in enumerate(retrieved_partial_forward_cache_list):
            outputs = self.model_manager.do_resume_forward(resume_at_layer=self.stop_at_layer, partial_forward_cache=partial_forward_cache)
            # cal query-loss & re-ranking 
            query_loss = self.model_manager.get_query_loss(encoded_query=query_ids, logits=outputs.logits[0, retrieved_chunk_len_list[i]-1: -1, :])
            del outputs
            torch.cuda.empty_cache()
            querylossList.append(query_loss)

        rerank_scores = np.sort(np.array(querylossList))[:self.topk]
        rerank_indices = np.argsort(np.array(querylossList))[:self.topk]
        retrieved_indices = [retrieve_map[i] for i in rerank_indices]
        context_score_dict = {retrieved_indice_context_dict[indice]: rerank_scores[i] for i, indice in enumerate(retrieved_indices)}
        # re-map and retrieve topk
        global_scores = [context_score_dict.get(context, 10000) for context in context_list]
        global_indices = np.argsort(np.array(global_scores))[: self.topk]
        return global_indices.tolist()

    def retrieve_by_query_loss_with_partial_forward(self, query: str, context_list: List[str], sep: str = '\n'):
        """
        step 1: partial forward to get key-value pairs in lower layer
        step 2: lower-layer retrieval with key-value pairs
        """
        unique_context_list = list(set(context_list))
        # get key-value pairs in middle layer
        for indice, context in enumerate(unique_context_list):
            # input text should be context + sep + query
            text = context + sep + query
            chunk_len = len(self.tokenizer.encode(context+sep))
            encoded_text = self.tokenizer(text, return_tensors='pt')
            input_ids = encoded_text.input_ids
            query_ids = input_ids[0, chunk_len:]
            partial_forward_cache = self.model_manager.do_partial_forward(text=text, stop_at_layer=self.stop_at_layer, output_last_attention=self.output_last_attention, \
                                                                          output_attentions=self.output_attentions, output_hidden_states=self.output_hidden_states)
            
            last_attention = partial_forward_cache.last_attention
            past_key_value = partial_forward_cache.past_key_values
            logits = partial_forward_cache.logits
            score = self.middle_layer_similarity(chunk_len=chunk_len, past_key_value=past_key_value, last_attention=last_attention, query_ids=query_ids, logits=logits)
            self.cache_manager.push(score, indice, chunk_len, partial_forward_cache)
        # get topk items
        topk_tuple = self.cache_manager.get_topk()
        unique_loss_list = [t[0] for t in topk_tuple]
        unique_indice_list = [t[1] for t in topk_tuple]
        chunk_len_list = [t[2] for t in topk_tuple]
        cache_list = [t[3] for t in topk_tuple]
        self.cache_manager.clear()
        
        unique_context_list = [unique_context_list[indice] for indice in unique_indice_list]
        context_loss_dict = dict(zip(unique_context_list, unique_loss_list))
        loss_list = [context_loss_dict[context] for context in context_list]
        global_indices = np.argsort(np.array(loss_list))

        return global_indices.tolist()

    def retrieve_and_generate(self, query, context_list, buffer_tokens, context_template, suffix=''):
        query_tokens = len(self.tokenizer.encode(query, add_special_tokens=False))
        entry = dict()
        if query_tokens > buffer_tokens:
            query = self.model_manager.truncate_by_tokens(query, buffer_tokens)
            entry['chunk_num'] = 0
            entry['ctxs'] = []
            entry['chunk_time'] = 0
            context = '' # query is too long
            prompt = context_template.format(context=context) + query + suffix
            start = time.time()
            entry['generated'] = self.model_manager.do_generate(prompt=prompt)[0]
            entry['generate_time'] = time.time() - start
        else:
            token_nums = buffer_tokens - query_tokens
            token_nums //= self.retriever_manager.topk # split to small chunks
            start = time.time()
            context_list = self.model_manager.split_long_context_to_chunk(text=context, token_nums=token_nums, task_name=self.task_name)
            entry['chunk_time'] = time.time() - start
            entry['chunk_num'] = len(context_list)
            ## retrieve
            start = time.time()
            retrieve_func = self.get_retrieve_func()
            # TODO: setting context_length
            context_list1 = [context_template.format(context=context) for context in context_list]
            ctxs = retrieve_func(query=query, context_list=context_list1)
            entry['retrieve_time'] = time.time() - start
            ctxs.reverse()
            entry['ctxs'] = ctxs
            # TODO: build cache
            ## inference
            query = query + suffix
            start = time.time()
            entry['generated'] = self.model_manager.do_generate_cache(query=query, cache=cache)[0]
            entry['generate_time'] = time.time() - start
        return entry

    # TODO
    def retrieve_and_generate_parallel(self, query, context_list, parallel_nums, buffer_tokens, context_template, suffix=''):
        query_tokens = len(self.tokenizer.encode(query, add_special_tokens=False))
        entry = dict()
        if query_tokens > buffer_tokens:
            query = self.model_manager.truncate_by_tokens(query, buffer_tokens)
            entry['chunk_num'] = 0
            entry['ctxs'] = []
            entry['chunk_time'] = 0
            context = '' # query is too long
            prompt = context_template.format(context=context) + query + suffix
            start = time.time()
            entry['generated'] = self.model_manager.do_generate(prompt=prompt)[0]
            entry['generate_time'] = time.time() - start
        else:
            token_nums = buffer_tokens - query_tokens
            token_nums //= self.retriever_manager.topk # split to small chunks
            start = time.time()
            context_list = self.model_manager.split_long_context_to_chunk(text=context, token_nums=token_nums, task_name=self.task_name)
            entry['chunk_time'] = time.time() - start
            entry['chunk_num'] = len(context_list)
            ## retrieve
            start = time.time()
            retrieve_func = self.get_retrieve_func()
            # TODO: setting context_length
            context_list1 = [context_template.format(context=context) for context in context_list]
            ctxs = retrieve_func(query=query, context_list=context_list1)
            entry['retrieve_time'] = time.time() - start
            ctxs.reverse()
            entry['ctxs'] = ctxs
            # TODO: build cache
            ## inference
            query = query + suffix
            start = time.time()
            entry['generated'] = self.model_manager.do_generate_cache_parallel(query=query, cache=cache)[0]
            entry['generate_time'] = time.time() - start
        return entry


