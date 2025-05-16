from typing import Optional, List, Dict, Tuple, Union
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
from src.models.modeling_llama_mbr3 import BaseModelOutputWithPastForMBR3
from vllm import LLM, SamplingParams
logger = logging.getLogger(__name__)
import os, time, json
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL2MAXLEN = json.load(open("src/models/model2maxlen.json", "r"))
MODEL2PATH = json.load(open("src/models/model2path.json", "r"))

def get_terminator(model_name, tokenizer):
    model_name = model_name.lower()
    if 'llama2' in model_name or 'llama-2' in model_name:
        return [tokenizer.eos_token_id]
    elif 'llama3' in model_name or 'llama-3' in model_name:
        return [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    else:
        raise ValueError("terminator not found")

def add_template(model_name, prompt):
    model_name = model_name.lower()
    if 'llama2' in model_name or 'llama-2' in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif 'llama3' in model_name or 'llama-3' in model_name:
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt


def ppl_generate(input_texts, model, tokenizer, choices_list, device=None):
    loss_list = []
    # here we set batch size to 1
    for choices, text in zip(choices_list, input_texts):
        filled_texts = [text+choice for choice in choices]
        loss = [_evaluate_loss(filled_text, model, tokenizer, device)[0] for filled_text in filled_texts]
        # inference one sentence per time
        loss_list.append(loss)
    lm_loss_list = np.array(loss_list)
    preds = lm_loss_list.argmin(axis=1).tolist()
    # print(preds)
    return preds


def _evaluate_loss(input_texts, model, tokenizer, device):
    with torch.no_grad():
        inputs = tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        # note here we assume padding is performed on the right, left padding token will affect position_id in gpt2
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
            shift_labels.size())
        ce_loss = loss.sum(-1).cpu().detach().numpy()  # -log(p(y))
        lens = (inputs["input_ids"] != tokenizer.pad_token_id).sum(-1).cpu().numpy()
    return ce_loss / lens


class ModelManager:
    def __init__(self, cfg) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.use_vllm = cfg.use_vllm
        self.model_name = cfg.model_name
        self.model_path = MODEL2PATH[cfg.model_name]
        self.model, self.tokenizer = self.load_model_tokenizer(self.model_path, use_vllm=self.use_vllm, self_define_lm=cfg.self_define_lm, device=self.device, gpu_memory_utilization=cfg.gpu_memory_utilization)
        self.n_tokens = MODEL2MAXLEN[cfg.model_name]
        self.terminators = get_terminator(self.model_name, self.tokenizer)
        self.generation_kwargs = OmegaConf.to_object(cfg.generation_kwargs)
            
        
    def truncate_by_tokens(self, input, max_tokens, manner: str = "middle"):
        tokens = self.tokenizer.encode(input)
        tokens = self.truncate_input(tokens, max_length=max_tokens, manner=manner)
        len_after = len(tokens)  # type: ignore
        assert len_after <= max_tokens
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def truncate_input(self, input: list, max_length: int, manner="middle"):
        if len(input) <= max_length:
            return input
        if manner == "middle":
            split = max_length // 2
            return input[0:split] + input[-split:]
        else:
            return None

    def load_model_tokenizer(self, model_path, use_vllm=False, self_define_lm=False, device='cuda:0', dtype=torch.float16, gpu_memory_utilization=0.5):
        logger.info(f'loading model from: {model_path}')
        if use_vllm:
            logger.info(f'Using vllm to speed up inference!')
            model = LLM(model=model_path, device=device, gpu_memory_utilization=gpu_memory_utilization)
            tokenizer = model.get_tokenizer()
        else:
            if 'llama' in model_path.lower():
                if self_define_lm:
                    logger.info('Using self-defined model loader.')
                    from src.models.modeling_llama_mbr3 import LlamaForCausalLM
                    model = LlamaForCausalLM.from_pretrained(model_path, device_map=device, torch_dtype=dtype)
                else:
                    from transformers import AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, torch_dtype=dtype)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device, trust_remote_code=True, torch_dtype=dtype)
            model = model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
        return model, tokenizer

    def get_model_tokenizer(self):
        return self.model, self.tokenizer

    def split_long_context_icl(self, text, token_nums, task_name):
        if task_name == 'trec':
            splitter = 'Question:'
        elif task_name == 'triviaqa':
            splitter = 'Passage'
        elif task_name == 'samsum':
            splitter = 'Dialogue'
        elif task_name in ['passage_count', 'passage_retrieval_en']:
            splitter = 'Paragraph'
        sentences = text.split(splitter)[1:]
        sentences = [splitter + sent for sent in sentences]
        chunks, chunk, chunk_len = [], '', 0
        for sent in sentences:
            chunk_len = len(self.tokenizer.encode(chunk + sent, add_special_tokens=False))
            if chunk_len < token_nums:
                chunk += sent
            else:
                if len(self.tokenizer.encode(chunk, add_special_tokens=False)) > token_nums:
                    chunk = self.truncate_by_tokens(chunk, token_nums)
                chunks.append(chunk)
                chunk = sent
        if chunk_len > 0:
            if len(self.tokenizer.encode(chunk, add_special_tokens=False)) > token_nums:
                chunk = self.truncate_by_tokens(chunk, token_nums)
            chunks.append(chunk)
        return chunks

    def split_long_context_to_chunk(self, text, token_nums, task_name, buffer=100):
        if task_name in ['trec', 'triviaqa', 'samsum', 'passage_count', 'passage_retrieval_en']:
            return self.split_long_context_icl(text, token_nums, task_name)
        if task_name == 'kv_retrieval':
            splitter = {','}
        elif task_name in ['code_debug', 'lcc', 'repobench-p']:
            splitter = {'\n'}
        else:
            splitter = {'\n', '.', '!', '?', ';'}
        encoded_text = self.tokenizer.encode(text, add_special_tokens=False)
        chunks =[]
        while len(encoded_text) > 0:
            truncated_ids = encoded_text[:token_nums]
            truncated_text = self.tokenizer.decode(truncated_ids)
            if len(truncated_ids) < token_nums:
                chunks.append(truncated_text)
                break
            else:
                counter = 0
                for i in range(len(truncated_text)-1, -1, -1):
                    if truncated_text[i] in splitter:
                        break
                    if counter >= buffer:
                        i = len(truncated_text)-1
                        break
                    counter += 1
                chunk = truncated_text[:i+1]
                if len(self.tokenizer.encode(chunk, add_special_tokens=False)) > token_nums:
                    chunk = self.truncate_by_tokens(chunk, token_nums)
                chunks.append(chunk)
                encoded_text = self.tokenizer.encode(truncated_text[i+1:], add_special_tokens=False) + encoded_text[token_nums:]

        return chunks

    # ABORTED
    def split_chunk(self, text, chunk_num, find_str = None):
        avg_token_nums = len(text) // chunk_num
        chunks = []
        front, rear = 0, 0
        idx = -1
        for i in range(1, chunk_num):
            if i*avg_token_nums <= rear:
                continue
            else:
                rear = i*avg_token_nums
                while text[rear] not in {'\n', '.', '!', '?', ';'} and rear < len(text):
                    rear += 1
                chunk = text[front: rear+1]
                front = rear + 1
            if find_str in chunk:
                idx = len(chunks)
            chunks.append(chunk)

        chunk = text[front:]
        if find_str in chunk:
            idx = len(chunks)
        chunks.append(chunk)

        if find_str:
            return chunks, idx
        else:
            return chunks

    # ABORTED
    def split_long_context_to_chunk1(self, text, token_nums, splitter='([\\n.!?;]\\s*)'):
        """
        Spliting the context into chunks of size chunk_size in complete sentences
        """
        if splitter != '([\\n.!?;]\\s*)':
            sentences = text.split(splitter)[1:]
            sentences = [splitter + sent for sent in sentences]
        else:
            while text[0] in {'\n', '.', '!', '?', ';'}:
                text = text[1:]
            while text[-1] not in {'\n', '.', '!', '?', ';'}:
                text = text[:-1]
            sents = re.split(splitter, text)
            sents = [sent for sent in sents if sent != '']
            sentences = []
            save = 0
            for i, sent in enumerate(sents):
                if save == i-1:
                    sentences[-1] += sent
                else:
                    sentences.append(sent)
                    save = i
                    
        chunks = []
        chunk = ''
        chunk_len = 0
        sentence_length = [len(self.tokenizer.encode(sent, add_special_tokens=False)) for sent in sentences]
        # tokenized length is not equal before and after concat
        for i, length in enumerate(sentence_length):
            if chunk_len < token_nums * 0.9 and chunk_len + length <= token_nums:
                chunk_len += length
                chunk += sentences[i]
            else:
                new_chunk = chunk + sentences[i]
                new_len = len(self.tokenizer.encode(new_chunk, add_special_tokens=False))
                if new_len <= token_nums:
                    chunk = new_chunk
                    chunk_len = new_len
                else:
                    chunks.append(chunk)
                    chunk = sentences[i]
                    chunk_len = length
        if chunk:
            chunks.append(chunk)
        # for sentence in sentences:
        #     new_chunk = chunk + sentence
        #     new_len = len(self.tokenizer.encode(new_chunk, add_special_tokens=False))
        #     if new_len <= token_nums:
        #         chunk = new_chunk
        #     else:
        #         if chunk:
        #             chunks.append(chunk)
        #         chunk = sentence
        # if chunk:
        #     chunks.append(chunk)
        return chunks

    def do_forward(self, input_ids: torch.Tensor = None, text: str = None, output_attentions=False, output_hidden_states=False, return_dict=True):
        if input_ids is None and text is not None:
            input_ids = self.tokenizer(text, return_tensors='pt').input_ids.to(self.model.device)
        elif text is None and input_ids is not None:
            input_ids = input_ids.to(self.model.device)
        else:
            raise Exception('input_ids and text cannot be both None')
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_hidden_states=output_hidden_states, output_attentions=output_attentions,  return_dict=return_dict)
        return outputs
        
    def do_partial_forward(self, input_ids: torch.Tensor, stop_at_layer: int, output_last_attention=False, output_attentions=False, output_hidden_states=False, use_cache=False, return_dict=True) -> BaseModelOutputWithPastForMBR3:
        input_ids = input_ids.to(self.model.device)
        with torch.no_grad():
            partial_forward_cache = self.model.partial_forward(input_ids=input_ids, stop_at_layer=stop_at_layer, \
                        output_last_attention=output_last_attention, output_attentions=output_attentions, \
                       output_hidden_states=output_hidden_states, use_cache=use_cache, return_dict=return_dict)
        return partial_forward_cache

    def do_resume_forward(self, partial_forward_cache: BaseModelOutputWithPastForMBR3, resume_at_layer: int, return_dict=True, output_attentions=False, output_hidden_states=False):
        with torch.no_grad():
            outputs = self.model.resume_forward(partial_forward_cache=partial_forward_cache, resume_at_layer=resume_at_layer, return_dict=return_dict)
        return outputs

    def get_query_loss(self, encoded_query: torch.Tensor, logits: torch.FloatTensor):
        encoded_query = encoded_query.to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        query_loss = loss_fn(logits, encoded_query)
        return query_loss.detach().cpu().item()

    def get_context_query_loss(self, query, context_list, sep = '\n'):
        """
        Calculate the self-information of the query given the context candidate
        """
        if not self.use_vllm:
            query_loss_list = []
            for candidate in context_list:
                encoded_input = self.tokenizer.encode(candidate + sep + query, return_tensors='pt')
                encoded_prefix = self.tokenizer.encode(candidate + sep)
                prefix_len = len(encoded_prefix)
                encoded_query = encoded_input[0,prefix_len:]
                encoded_query_len = len(encoded_query)
                output = self.do_forward(encoded_input)
                output_logprobs = output.logits[0]
                query_loss_list.append(self.get_query_loss(encoded_query, output_logprobs[prefix_len-1:-1]))
            return query_loss_list
        else:
            query_loss_list = []
            for candidate in context_list:
                encoded_input = self.tokenizer.encode(candidate + sep + query)
                encoded_prefix = self.tokenizer.encode(candidate + sep)
                prefix_len = len(encoded_prefix)
                encoded_query = encoded_input[prefix_len:]
                encoded_query_len = len(encoded_query)
                sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=1, prompt_logprobs=1)
                output_logprobs = self.model.generate(prompt_token_ids=[encoded_input], sampling_params=sampling_params, use_tqdm=False)[0].prompt_logprobs
                query_loss = 0
                for indice, logprob in zip(encoded_query, output_logprobs[prefix_len:]):
                    query_loss += -logprob[indice].logprob
                query_loss /= encoded_query_len
                query_loss_list.append(query_loss)
            return query_loss_list

    def do_generate(self, prompt: str):
        # encode
        tokenized_prompt = self.tokenizer.encode_plus(prompt, truncation=False, return_tensors='pt', add_special_tokens=False).to(self.device)
        total_length = tokenized_prompt.input_ids.shape[1] + self.generation_kwargs['max_new_tokens']
        assert total_length < self.n_tokens, (f"total length {total_length} >= max length {self.n_tokens}!")
        if self.use_vllm:
            sampling_params = SamplingParams(temperature=self.generation_kwargs['temperature'], n=self.generation_kwargs['num_return_sequences'], max_tokens=self.generation_kwargs['max_new_tokens'])
            outputs = self.model.generate(prompts=prompt, sampling_params=sampling_params, use_tqdm=False)
            generated = [output.text for output in outputs[0].outputs]
        else:
            with torch.no_grad():
                outputs = self.model.generate(input_ids=tokenized_prompt.input_ids,
                                                attention_mask=tokenized_prompt.attention_mask,
                                                eos_token_id=self.terminators,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                **self.generation_kwargs)
            prompt_len = int(tokenized_prompt['attention_mask'].shape[1])
            generated = [self.tokenizer.decode(outputs[i, prompt_len:], skip_special_tokens=True) for i in range(outputs.shape[0])]
        return generated
    
    def do_generate_cache(self, query, cache):
        pass

    def do_generate_cache_parallel(self, query, cache):
        pass

