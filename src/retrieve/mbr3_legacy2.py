from typing import List
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from functools import partial
# config
# from src.retrieve.self_info_utils import *
import gc
from tqdm import tqdm
from loguru import logger
from src.utils.splitter import split_long_sentence, regex
from src.models.modeling_llama_mbr3 import BaseModelOutputWithPastForMBR3
def split_chunk(context: str, chunk_size: int):
    """
    Spliting the context into chunks of size chunk_size in complete sentences
    """
    chunks = split_long_sentence(context, regex, chunk_size=chunk_size)
    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


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

class MBR3:
    def __init__(self, 
            method: str = 'mbr3',
            infer_model = None, 
            infer_tokenizer = None,
            n_tokens: int = 4096,
            sep: str = '\n',
            topk: int = 7,
            retrieve_topk: int = 12,
            metric: str = 'cosine',
            chunk_size: int = '200',
            embedder_type: str = 'bert',
            retrieve_level: str = 'chunk',
            embed_query_type: str = 'cls',
            llm_embed_type = 'hidden_state',
            use_vllm = False,
            layer=12):
    
        self.infer_model = infer_model
        self.infer_tokenizer = infer_tokenizer
        self.topk = topk,
        self.retrieve_topk = retrieve_topk
        self.n_tokens = n_tokens
        self.metric = metric
        self.chunk_size = chunk_size
        self.embedder_type = embedder_type
        self.layer = layer,
        self.retrieve_level = retrieve_level
        self.embed_query_type = embed_query_type
        self.llm_embed_type = llm_embed_type
        self.use_vllm = use_vllm
        self.sep = sep
        self.method = method
        logger.info(f'Using {self.method} to retrieve')

    def retrieve_in_middle_layer(self, query: str, partial_forward_cache_list: List[BaseModelOutputWithPastForMBR3], topk: int, layer: int = -1):
        """
        retrieve with criteria like k, v similarity in the middle layer
        return: topk kvPairs, topk indices
        """
        encoded_query = self.infer_tokenizer.encode(query)
        query_len = len(encoded_query) - 1 # remove <bos>
        scores = []

        # calculate similarity between query and kvPair
        # metrics: k*k^T, v*v^T, attention_score
        # get: similarity score
        for partial_forward_cache in partial_forward_cache_list:
            last_hidden_states = partial_forward_cache.hidden_states
            attentions = partial_forward_cache.attentions
            past_key_value = partial_forward_cache.past_key_values
            score = self.middle_layer_similarity(query_len, past_key_value=past_key_value, last_hidden_states=last_hidden_states, attentions=attentions, layer = layer) # type: ignore
            # print(score.shape)
            scores.append(score)
        indices = topk_chunk_in_token_level(scores, topk)

        return [partial_forward_cache_list[indice] for indice in indices], indices

    @staticmethod
    def middle_layer_similarity(query_len: int, past_key_value: Optional[List[dict[str, torch.FloatTensor]]], last_hidden_states: Optional[torch.FloatTensor], attentions: List[torch.FloatTensor], use_key: bool = True, layer: int = -1):
        """
        calculate similarity between query and kvPair
        selecting metrics: k*k^T, v*v^T, attention_score
        return: similarity score
        """
        # use key as default
        if use_key:
            key_tensor = past_key_value[-1][0]
            # remove <bos>
            key_tensor = key_tensor[:, :, 1:, :]
            # split away query and context
            chunk_tensor = key_tensor[:, :, :-query_len, :]
            query_tensor = key_tensor[:, :, -query_len:, :]
        else:
            value_tensor = past_key_value[-1][1]
            value_tensor = value_tensor[:, :, 1:, :]
            chunk_tensor = value_tensor[:, :, :-query_len, :]
            query_tensor = value_tensor[:, :, -query_len:, :]

        # calculate similarity
        sim_scores = torch.matmul(chunk_tensor, query_tensor.transpose(-1, -2))
        # print(sim_scores.shape)
        # pooling over all heads
        sim_scores = torch.mean(sim_scores, dim=1)
        # pooling over all query tokens
        sim_scores = torch.mean(sim_scores, dim=-1)
        # print(sim_scores.shape)
        # print('='*20)
        return sim_scores


    def retrieve_by_query_loss(self, query: str, context_list: List[str], sep=None, n_tokens=None, retrieve_topk=None, topk=None, partial_retrieval=False, stop_at_layer=10):
        if topk is None:
            raise ValueError('topk is None, please set topk')
        n_tokens = self.n_tokens if n_tokens is None else n_tokens
        sep = '\n' if sep is None else sep
        if self.use_vllm:
            loss = self.cal_query_loss_vllm(query, context_list, sep=sep, n_tokens=n_tokens)
        elif partial_retrieval:
            if retrieve_topk is None:
                retrieve_topk = self.retrieve_topk
                logger.info(f'Using {retrieve_topk} to retrieve topk chunks in middle layer')
            return self.retrieve_by_query_loss_with_partial_forward(query, context_list, sep=sep, n_tokens=n_tokens, \
                            topk=topk, stop_at_layer=stop_at_layer, retrieve_topk=retrieve_topk)
        else:
            loss = self.cal_query_loss_origin(query, context_list, sep=sep, n_tokens=n_tokens)
            # loss = self.cal_all_loss_ablation(query, context_list, sep=sep, n_tokens=n_tokens)
            # loss = self.cal_chunk_loss_ablation(query, context_list, sep=sep, n_tokens=n_tokens)
        global_indices = np.argsort(np.array(loss))[:topk]
        return global_indices.tolist()

    def cal_query_loss_origin(self, query: str, context_list: List[str], sep = '\n', n_tokens=4096):
        """
        Calculate the self-information of the query given the context candidate
        """
        query_loss_list = []
        model = self.infer_model
        tokenizer = self.infer_tokenizer
        for candidate in context_list:
            encoded_input = tokenizer.encode(candidate + sep + query, return_tensors='pt')
            if encoded_input.shape[1] +1 >= n_tokens:
                diff = encoded_input.shape[1] - n_tokens -1
                tmp = tokenizer.encode(candidate)[:diff]
                candidate  = tokenizer.decode(tmp)
                encoded_input = tokenizer.encode(candidate + sep + query, return_tensors='pt')
            encoded_prefix = tokenizer.encode(candidate + sep)
            prefix_len = len(encoded_prefix)
            encoded_query = encoded_input[0,prefix_len:]
            encoded_query_len = len(encoded_query)
            output = model(torch.tensor(encoded_input).to(model.device), return_dict=True)
            output_logprobs = output.logits[0]
            query_loss = 0
            # TODO: make sure logprob is before softmax in transforerms and vllm
            for indice, logprob in zip(encoded_query, output_logprobs[prefix_len:]):
                query_loss += -logprob[indice].item()
            query_loss /= encoded_query_len
            query_loss_list.append(query_loss)
        return query_loss_list
        
    def cal_query_loss_vllm(self, query: str, context_list: List[str], sep = '\n', n_tokens=4096):
        """
        Calculate the self-information of the query given the context candidate
        """
        from vllm import SamplingParams
        query_loss_list = []
        tokenizer = self.infer_model.get_tokenizer()
        for candidate in context_list:
            encoded_input = tokenizer.encode(candidate + sep + query)
            if len(encoded_input) +1 >= n_tokens:
                diff = len(encoded_input) - n_tokens -1
                tmp = tokenizer.encode(candidate)[:diff]
                candidate  = tokenizer.decode(tmp)
                encoded_input = tokenizer.encode(candidate + sep + query)
            encoded_prefix = tokenizer.encode(candidate + sep)
            prefix_len = len(encoded_prefix)
            encoded_query = encoded_input[prefix_len:]
            encoded_query_len = len(encoded_query)
            sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=1, prompt_logprobs=1)
            output_logprobs = self.infer_model.generate(candidate + sep + query, sampling_params, use_tqdm=False)[0].prompt_logprobs
            query_loss = 0
            for indice, logprob in zip(encoded_query, output_logprobs[prefix_len:]):
                query_loss += -logprob[indice].logprob
            query_loss /= encoded_query_len
            query_loss_list.append(query_loss)
        return query_loss_list
    

    def retrieve_by_query_loss_with_partial_forward(self, query: str, context_list: List[str], stop_at_layer: int, topk: int, retrieve_topk: int, n_tokens : int = 4096, sep: str = '\n'):
        """
        step 1: partial forward to get key-value pairs in lower layer
        step 2: lower-layer retrieval with key-value pairs
        step 3: resume forward to the rest layers to get the final logits
        step 4: re-rank with the retrieved chunk
        """
        # TODO: change to batch version
        # TODO: check utilize of n_tokens
        partial_forward_cache_list = []
        outputList = []
        querylossList = []
        # get key-value pairs in middle layer
        for context in context_list:
            input_text = context + self.sep + query
            encoded_text = self.infer_tokenizer(input_text, return_tensors='pt')
            input_ids = encoded_text.input_ids.to(self.infer_model.device)
            with torch.no_grad():
                partial_forward_cache = self.infer_model.partial_forward(input_ids=input_ids, stop_at_layer=stop_at_layer, return_dict=True)
                partial_forward_cache_list.append(partial_forward_cache)
        # get topk indices
        partial_forward_cache_list, retrieved_inds = self.retrieve_in_middle_layer(query=query, partial_forward_cache_list=partial_forward_cache_list, topk=retrieve_topk, layer=stop_at_layer)
        retrieve_map = dict(zip(range(len(retrieved_inds)), retrieved_inds))
        # continue forward to the rest layers
        for partial_forward_cache in partial_forward_cache_list:
            with torch.no_grad():
                outputs = self.infer_model.resume_forward(resume_at_layer=stop_at_layer, partial_forward_cache=partial_forward_cache, return_dict=True)
            outputList.append(outputs)
        # cal query-loss & re-ranking 
        for outputs in outputList:
            query_loss = self.cal_query_loss_with_outputs(outputs.logits, query)
            querylossList.append(query_loss)
        rerank_inds = np.argsort(np.array(querylossList))[:topk]
        global_inds = [retrieve_map[i] for i in rerank_inds]
        # return topk index
        return global_inds

    def cal_query_loss_with_outputs(self, logits, query: str):
        """
        calculate the loss of query tokens in the logits, query tokens is behind the chunk tokens with the format of <chunk> <query>
        return: loss
        """
        encoded_query = self.infer_tokenizer(query).input_ids[1:] # remove <bos> token
        query_len = len(encoded_query)
        query_loss = 0
        for i, qtoken in enumerate(encoded_query):
            logit = logits[0, -(query_len - i), :]
            query_loss += logit[qtoken]
        return query_loss.cpu().detach().numpy()/query_len


    def cal_all_loss_ablation(self, query: str, context_list: List[str], sep = '\n', n_tokens=4096):
        """
        Calculate the self-information of the query given the context candidate
        """
        query_loss_list = []
        model = self.infer_model
        tokenizer = self.infer_tokenizer
        for candidate in context_list:
            encoded_input = tokenizer.encode(candidate + sep + query, return_tensors='pt')
            if encoded_input.shape[1] +1 >= n_tokens:
                diff = encoded_input.shape[1] - n_tokens -1
                tmp = tokenizer.encode(candidate)[:diff]
                candidate  = tokenizer.decode(tmp)
                encoded_input = tokenizer.encode(candidate + sep + query, return_tensors='pt')
            encoded_input_len = len(encoded_input[0])
            with torch.no_grad():
                output = model(torch.tensor(encoded_input).to(model.device), return_dict=True)
            
            output_logprobs = output.logits[0]
            query_loss = 0
            for indice, logprob in zip(encoded_input[0], output_logprobs):
                query_loss += -logprob[indice].item()
            query_loss /= encoded_input_len
            query_loss_list.append(query_loss)
        return query_loss_list



    def cal_chunk_loss_ablation(self, query: str, context_list: List[str], sep = '\n', n_tokens=4096):
        """
        Calculate the self-information of the query given the context candidate
        """
        chunk_loss_list = []
        model = self.infer_model
        tokenizer = self.infer_tokenizer
        for candidate in context_list:
            if len(candidate) == 0:
                continue
            encoded_input = tokenizer.encode(query + sep + candidate, return_tensors='pt')
            if encoded_input.shape[1] +1 >= n_tokens:
                diff = encoded_input.shape[1] - n_tokens + 1
                tmp = tokenizer.encode(candidate)[:diff]
                candidate  = tokenizer.decode(tmp)
                encoded_input = tokenizer.encode(query + sep + candidate, return_tensors='pt')
            encoded_prefix = tokenizer.encode(query + sep)
            prefix_len = len(encoded_prefix)
            encoded_chunk = encoded_input[:,prefix_len:]
            encoded_chunk_len = encoded_chunk.shape[1]
            with torch.no_grad():
                output = model(torch.tensor(encoded_input).to(model.device), return_dict=True)
            output_logprobs = output.logits[0]
            chunk_loss = 0
            for indice, logprob in zip(encoded_chunk[0], output_logprobs[prefix_len:]):
                chunk_loss += -logprob[indice].item()
            chunk_loss /= encoded_chunk_len
            chunk_loss_list.append(chunk_loss)
        return chunk_loss_list
        