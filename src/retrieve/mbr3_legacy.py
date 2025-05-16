from typing import List
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from functools import partial
from src.utils.splitter import split_long_sentence, regex
# config
from src.retrieve.self_info_utils import *
import gc
from tqdm import tqdm
from loguru import logger

def split_chunk(context: str, chunk_size: int):
    """
    Spliting the context into chunks of size chunk_size in complete sentences
    """
    chunks = split_long_sentence(context, regex, chunk_size=chunk_size)
    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class MBR3:
    def __init__(
            self,
            topk: int = 5,
            device: str = 'cuda:0',
            metric: str = 'cosine',
            chunk_size: int = '200',
            method: str = 'embedder',
            embedder_type: str = 'sentence-bert',
            embedder_path: str = '',
            retrieve_level: str = 'chunk',
            embed_query_type: str = 'cls',
            infer_model = None,
            infer_tokenizer = None,
            llm_embed_type = 'hidden_state',
            layer=20,
            use_vllm=False):
        
        self.topk = topk
        self.device = device
        self.metric = metric
        self.chunk_size = chunk_size
        self.method = method
        self.embedder_type = embedder_type
        # self.retrieve_level = retrieve_level
        self.embed_query_type = embed_query_type
        self.embedder_path = embedder_path
        self.llm_embed_type = llm_embed_type
        self.layer = layer
        self.use_vllm = use_vllm
        if self.method == 'rerank_mbr3':
            logger.info(f'use layer: {layer} for retrieval')
        # config embedder
        # logger.info(f'config method: {method}')
        if 'mbr3' in self.method and infer_model is not None:
            self.infer_model = infer_model
            self.infer_tokenizer = infer_tokenizer
        if self.embedder_path != '':
            self._config_embedder(embedder_type, embedder_path, retrieve_level)

    def _config_embedder(self, embedder_type: str, embedder_path: str, retrieve_level: str = 'chunk'):
        # config embedding model
        if retrieve_level == 'token' and embedder_type != 'bert':
            raise ValueError("Token-level retrieval only support bert")
        if self.embedder_type == 'bert':
            # max_length = 512, embed_dim = 768
            logger.info('loading bert-base-uncased')
            embed_tokenizer = AutoTokenizer.from_pretrained(embedder_path)
            embed_model = AutoModel.from_pretrained(embedder_path, device_map=self.device)
            embed_model = embed_model.eval()
        elif self.embedder_type == 'sentence-bert':
            # max_length = 256, embed_dim = 384
            logger.info('loading sentence-bert')
            embed_tokenizer = AutoTokenizer.from_pretrained(embedder_path)
            embed_model = AutoModel.from_pretrained(embedder_path).to(self.device)
            embed_model = embed_model.eval()
        elif self.embedder_type == 'llm':
            logger.info('loading transformers llm: ', embedder_path)
            embed_tokenizer = AutoTokenizer.from_pretrained(embedder_path)
            embed_model = AutoModel.from_pretrained(embedder_path, torch_dtype=torch.bfloat16).to(self.device)
            embed_model = embed_model.eval()
        else:
            raise ValueError("Invalid embedding model")
        self.embed_model = embed_model
        self.embed_tokenizer = embed_tokenizer
        logger.info(f'finish config embedder: {embedder_type}')

    def embed_text(self, text: str, embedder_type: str= 'llm', llm_embed_type: str= 'key', normalize: bool = False, layer: int = -1):
        embedder_type = embedder_type if embedder_type != '' else self.embedder_type
        llm_embed_type = llm_embed_type if llm_embed_type != '' else self.llm_embed_type
        layer = layer if layer != -1 else self.layer
        if embedder_type == 'bert':
            embed_func = partial(self.embed_text_bert, normalize=normalize)
        elif embedder_type == 'sentence-bert':
            embed_func = partial(self.embed_text_sb, normalize=normalize)
        elif embedder_type == 'llm':
            if llm_embed_type == 'hidden_state':
                embed_func = partial(self.embed_text_llm_hidden_state, layer=layer)
                # [bs, seq_len, embed_dim]
            elif llm_embed_type == 'attention_score':
                embed_func = partial(self.embed_text_llm_attention_score, layer=layer)
                # [bs, n_heads, seq_len, seq_len]
            elif llm_embed_type == 'query':
                embed_func = partial(self.embed_text_llm_query, layer=layer)
                # [bs, seq_len, embed_dim]
            elif llm_embed_type == 'key':
                embed_func = partial(self.embed_text_llm_key, layer=layer)
                # [bs, seq_len, embed_dim]
            else:
                raise ValueError("Invalid llm_embed_type")
        else:
            raise ValueError("Invalid embedder type")
        return embed_func(text)

    def embed_text_bert(self, text: str, normalize: bool = False):
        tokenized_query = self.embed_tokenizer(text, max_length=512, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            embed_query = self.embed_model(**tokenized_query).last_hidden_state
        if self.embed_query_type == 'cls':
            embed_query = embed_query[:, 0]
        else:
            if self.embed_query_type == 'mean':
                embed_query = torch.mean(embed_query, dim=1)
            elif self.embed_query_type == 'max':
                embed_query = embed_query[:, 1:-1]
                embed_query = torch.max(embed_query, dim=1).values
            else:
                raise ValueError("Invalid embed_query_type")
        if normalize:
            embed_query = F.normalize(embed_query, p=2, dim=-1)
        return embed_query[0]
    
    def embed_text_sb(self, text: str, normalize: bool = False):
        tokenized_query = self.embed_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,      # 显式启用截断
            max_length=512,       # 设置最大长度为 512
            padding='max_length'  # 可选：填充到 512
        ).to(self.device)
        with torch.no_grad():
            embed_query = self.embed_model(**tokenized_query).last_hidden_state
        embed_query = mean_pooling(embed_query, tokenized_query['attention_mask'])
        if normalize:
            embed_query = F.normalize(embed_query, p=2, dim=-1)
        return embed_query

    def embed_text_llm_hidden_state(self, text: str, layer: int = 13):
        tokenized_query = self.embed_tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            embed_query = self.embed_model(**tokenized_query, output_hidden_states = True).hidden_states[layer]
        # print(embed_query.shape)
        embed_query = torch.mean(embed_query, dim=1)
        return embed_query[0]
    
    def embed_text_llm_hidden_state(self, text: str, layer: int = 13):
        tokenized_query = self.embed_tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            attention_score = self.embed_model(**tokenized_query, output_attentions = True).attentions[layer]
            # [bs, n_heads, seq_len, seq_len]
        embed_query = torch.mean(attention_score, dim=1)
        return embed_query[0]

    def cal_key_quey_cache(self, text: str):
        tokenized_query = self.embed_tokenizer(text, return_tensors='pt').to(self.device)
        queries = []
        keys = []
        def get_attention_scores_hook(layer_idx):
            def hook(module, input, output):
                # Access the hidden states from the model
                hidden_states = output[0]  # Get hidden states from the output
                
                # Apply q_proj and k_proj directly to the hidden states
                query_states = module.q_proj(hidden_states)
                key_states = module.k_proj(hidden_states)
                
                queries.append((layer_idx, query_states.detach().clone()))
                keys.append((layer_idx, key_states.detach().clone()))
            return hook
        hooks = []
        for i, layer in enumerate(self.embed_model.layers):
            # Register hook to the attention module
            hook = layer.self_attn.register_forward_hook(get_attention_scores_hook(i))
            hooks.append(hook)
        # 进行推理
        with torch.no_grad():
            outputs = self.embed_model(**tokenized_query)
        # 移除钩子以避免内存泄漏
        for hook in hooks:
            hook.remove()
        # 排序（确保按层顺序显示）
        sorted_queries = sorted(queries, key=lambda x: x[0])
        sorted_keys = sorted(keys, key=lambda x: x[0])
        sorted_queries = [query[1] for query in sorted_queries]
        sorted_keys = [key[1] for key in sorted_keys]
        return sorted_queries, sorted_keys

    def embed_text_llm_query(self, text: str, layer: int = 24):
        queries, _ = self.cal_key_quey_cache(text)
        # 获取query表征并处理
        query_repr = queries[layer]
        # 对所有token位置的query进行平均池化
        embed_query = torch.mean(query_repr, dim=1)
        return embed_query[0]
    
    def embed_text_llm_key(self, text: str, layer: int = 13):
        _, keys = self.cal_key_quey_cache(text)
        # 获取query表征并处理
        key_repr = keys[layer]
        # 对所有token位置的query进行平均池化
        embed_key = torch.mean(key_repr, dim=1)
        return embed_key[0]

    def token_level_retrieve_embedder(self, query: str, context_list: List[str], topk: int = 5, normalize: bool = False, order=False):
        """
        Retrieve the most similar k context candidates based on token-level similarity in embedding space
        """
        if len(context_list) <= topk:
            return list(range(len(context_list)))
        assert self.embedder_type == 'bert', "token-level retrieval only support bert-base-uncased"
        # embed query
        embed_query = self.embed_text(query, normalize=True)
        # embed candidates
        embed_context_list = [self.embed_text(candidate, normalize=normalize) for candidate in context_list]
        embed_context_array = torch.stack(embed_context_list)
        group_indices = np.array([item.shape[0] for item in embed_context_list])
        group_indices = np.cumsum(group_indices)
        # calculate similarity
        if self.metric == 'cosine':
            embed_context_array = F.cosine_similarity(embed_query, embed_context_array)
            _, token_indices = torch.topk(embed_context_array, topk)
        elif self.metric == 'euclidean':
            distances = torch.cdist(embed_query, embed_context_array, p=2)
            _, token_indices = torch.topk(distances, topk, dim=-1)
        else:
            raise ValueError("Invalid metric")
        # convert to global indices
        global_indices = []
        for token_indice in token_indices:
            global_indice = 0
            for i, maxind in enumerate(group_indices):
                if token_indice >= maxind:
                    global_indice = i+1
                else:
                    break
            global_indices.append(global_indice)
        global_indices = list(set(global_indices))
        if order:
            global_indices.sort()
            topk_indices = global_indices
        assert len(topk_indices) == topk
        return topk_indices

    def token_level_retrieve_embedder(self, query: str, context_list: List[str], topk: int = 5, order=False):
        """
        Retrieve the most similar k context candidates based on token-level similarity in embedding space
        """
        if len(context_list) <= topk:
            return list(range(len(context_list)))
        # embed query
        embed_query = self.embed_text(query, embedder_type='llm', llm_embed_type='key', normalize=True, layer=self.layer)
        # embed candidates        
        embed_context_list = [self.embed_text(candidate, embedder_type='llm', llm_embed_type='key', normalize=True, layer=self.layer) for candidate in context_list]
        embed_context_len_list = [candidate.shape[1] for candidate in embed_context_list]
        cumsum_len_list = np.cumsum(embed_context_len_list)
        embed_context_array = torch.stack(embed_context_list)
        attention_scores = torch.matmul(embed_query, embed_context_array.transpose(0, 1))
        chunk_token_scores = attention_scores.mean(dim=-1)
        # TODO: topk token num -> topk chunk num
        topk_indices = chunk2token(chunk_token_scores[0], topk)
        topk_indices = torch.topk(chunk_token_scores[0], topk).indices
        if order:
            topk_indices = topk_indices.sort().values
        topk_indices = topk_indices.tolist()
        assert len(topk_indices) == topk
        return topk_indices

    def chunk_level_retrieve_embedder(self, query: str, context_list: List[str], topk: int = 5, order=False):
        """
        Retrieve the most similar k context candidates based on chunk-level similarity in embedding space
        """
        if len(context_list) <= topk:
            return list(range(len(context_list)))
        # embed query
        embed_query = self.embed_text(query, embedder_type=self.embedder_type, normalize=True, layer=self.layer)
        # embed candidates        
        embed_context_list = [self.embed_text(candidate, embedder_type=self.embedder_type, normalize=True, layer=self.layer)[0] for candidate in context_list]
        embed_context_array = torch.stack(embed_context_list)

        # Compute cosine similarities
        # similarities = F.cosine_similarity(
        #     embed_query.unsqueeze(0),
        #     embed_context_array,
        #     dim=1
        # )
        similarities = torch.matmul(embed_query, embed_context_array.transpose(0, 1))[0]
        # Get topk indices and scores
        _, topk_indices = torch.topk(similarities, k=min(topk, len(context_list)))
        if order:
            topk_indices = topk_indices.sort().values
        topk_indices = topk_indices.tolist()
        assert len(topk_indices) == topk
        return topk_indices

    def chunk_level_retrieve_query_loss(self, query: str, context_list: List[str], topk: int = 5, n_tokens=4096, order=False):
        """
        Retrieve the most similar k context candidates based on self-information of the query given the context candidate
        """
        if len(context_list) <= topk:
            return list(range(len(context_list)))
        if self.use_vllm:
            query_loss = self.cal_query_loss_vllm(query, context_list, n_tokens=n_tokens)
        else:
            query_loss = self.cal_query_loss_origin(query, context_list, n_tokens=n_tokens)
        query_loss = np.array(query_loss)
        topk_indices = query_loss.argsort()[:topk]
        if order:
            topk_indices = np.sort(topk_indices)
        topk_indices = topk_indices.tolist()
        assert len(topk_indices) == topk
        return topk_indices
    
    def cal_query_loss_optim(self, query: str, context_list: List[str], sep = '\n', n_tokens=4096):
        """
        Calculate the self-information of the query given the context candidate
        """
        from vllm import SamplingParams
        tokenizer = self.infer_model.get_tokenizer()
        sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=1, prompt_logprobs=1)

        prefix_len_list = []
        query_loss_list = []
        encoded_query_list = []
        encoded_query_len_list = []
        truncated_candidate_list = []

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

            prefix_len_list.append(prefix_len)
            truncated_candidate_list.append(candidate)
            encoded_query_list.append(encoded_query)
            encoded_query_len_list.append(encoded_query_len)

        outputs = self.infer_model.generate([candidate + sep + query for candidate in truncated_candidate_list], sampling_params, use_tqdm=False)
        for encoded_query, output, prefix_len, encode_query_len in zip(encoded_query_list, outputs, prefix_len_list, encoded_query_len_list):
            query_loss = 0
            output_logprobs=output.prompt_logprobs
            for indice, logprob in zip(encoded_query, output_logprobs[prefix_len:]):
                query_loss += -logprob[indice].logprob
            query_loss /= encoded_query_len
            query_loss_list.append(query_loss)
    
        return query_loss_list
        
        
    def cal_query_loss_origin(self, query: str, context_list: List[str], sep = '\n', n_tokens=4096):
        """
        Calculate the self-information of the query given the context candidate
        """
        query_loss_list = []
        model = self.infer_model
        tokenizer = self.infer_tokenizer
        for candidate in context_list:
            encoded_input = tokenizer.encode(candidate + sep + query, return_tensors='pt')
            if len(encoded_input) +1 >= n_tokens:
                diff = len(encoded_input) - n_tokens -1
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
        # with torch.no_grad():
        #     for input in tqdm(candidate_query_input, desc='encoding'):
        #         output = llm(**input)
        #         output_logit_list.append(output.logits)
        #         del output
        #         torch.cuda.empty_cache()
        #         gc.collect()
        # query_loss_list = [self.nll_loss(input, output_logits, encoded_query_len) for input, output_logits in tqdm(zip(candidate_query_input, output_logit_list), desc='calculate nll loss')]
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
        # with torch.no_grad():
        #     for input in tqdm(candidate_query_input, desc='encoding'):
        #         output = llm(**input)
        #         output_logit_list.append(output.logits)
        #         del output
        #         torch.cuda.empty_cache()
        #         gc.collect()
        # query_loss_list = [self.nll_loss(input, output_logits, encoded_query_len) for input, output_logits in tqdm(zip(candidate_query_input, output_logit_list), desc='calculate nll loss')]
        return query_loss_list
        
    ## abort
    # def chunk_level_retrieve_chunk_loss(self, query: str, context_list: List[str], topk: int = 5, llm=None, n_tokens=4096, order=False):
    #     """
    #     Retrieve the most similar k context candidates based on self-information of the query given the context candidate
    #     """
    #     if len(context_list) <= topk:
    #         return list(range(len(context_list)))
    #     if llm is None :
    #         raise ValueError("llm must be provided")
    #     query_loss = self.cal_chunk_loss(query, context_list, llm, n_tokens=n_tokens)
    #     query_loss = np.array(query_loss)
    #     topk_indices = query_loss.argsort()[:topk]
    #     if order:
    #         topk_indices = np.sort(topk_indices)
    #     topk_indices = topk_indices.tolist()
    #     return topk_indices
    
    # def cal_chunk_loss(self, query: str, context_list: List[str], llm, sep = '\n', n_tokens=4096):
    #     """
    #     Calculate the self-information of the query given the context candidate
    #     """
    #     from vllm import SamplingParams
    #     query_loss_list = []
    #     tokenizer = llm.get_tokenizer()
    #     for candidate in context_list:
    #         encoded_input = tokenizer.encode(query + sep + candidate)
    #         if len(encoded_input) +1 >= n_tokens:
    #             encoded_input = encoded_input[:n_tokens -1]
    #         encoded_prefix = tokenizer.encode(query + sep)
    #         prefix_len = len(encoded_prefix)
    #         encoded_candidate = encoded_input[prefix_len:]
    #         encoded_candidate_len = len(encoded_candidate)
    #         sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=1, prompt_logprobs=1)
    #         output_logprobs = llm.generate(query + sep + candidate, sampling_params, use_tqdm=False)[0].prompt_logprobs
    #         query_loss = 0
    #         for indice, logprob in zip(encoded_candidate, output_logprobs[prefix_len:]):
    #             query_loss += -logprob[indice].logprob
    #         query_loss /= encoded_candidate_len
    #         query_loss_list.append(query_loss)
    #     return query_loss_list
    # 
    # def nll_loss(self, entry, output_logits, suffix_len):
    #     shift_logits = output_logits[0, -suffix_len:, :].contiguous()
    #     shift_labels = entry.input_ids[0, -(1+suffix_len):-1].contiguous()
    #     log_probs = shift_logits.log_softmax(dim=-1)
    #     criterion = torch.nn.NLLLoss()
    #     # print('logit shape:', shift_logits.shape)
    #     # print('label shape:', shift_labels.shape)
    #     loss = criterion(log_probs, shift_labels)
    #     return loss.detach().cpu().numpy().item()

