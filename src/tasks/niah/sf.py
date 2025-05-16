from src.utils.misc import levenshtein_distance_word_level
from typing import List, Union
from src.utils.misc import load_jsonl
import time
from src.tasks.niah.config import NEEDLE

def build_niah_prompt(encoded_context: List[int], context_length: int, depth: float, tokenizer, sep=' ') -> str:
    tokenized_sep = tokenizer.encode(sep, add_special_tokens=False)
    # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
    period_tokens = tokenizer.encode('.', add_special_tokens=False)
    # config actual length
    percent = depth * 100
    ## step 1: load PaulGrahamEssay and process
    ## step 2: Truncate the haystack dir essays to the context length you desire
    # encoded_context = tokenizer.encode(context)
    # repeat = context_length // len(encoded_context) +1
    # context = '\n'.join([doc['text'] for doc in document_list] * repeat)
    context = tokenizer.decode(encoded_context[:context_length],skip_special_tokens=True)
    ## step 3: Insert your random statement according to your depth percent
    tokens_needle = tokenizer.encode(NEEDLE)
    tokens_context = tokenizer.encode(context)
    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) + 2 > context_length:
        tokens_context = tokens_context[:context_length - len(tokens_needle) -2]
    if percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        if tokenizer.decode(tokens_context[-5:], skip_special_tokens=True)[-1] not in {'\n', '.', '!', '?', ';'}:
            tokens_context += period_tokens
        tokens_new_context = tokens_context + tokenized_sep + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (percent / 100))
        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]
        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]
        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokenized_sep + tokens_needle + tokenized_sep + tokens_context[insertion_point:]
    # Convert back to a string and return it
    new_context = tokenizer.decode(tokens_new_context,skip_special_tokens=True)
    return new_context


def evaluate_niah(prompt, answer, tokenizer, model, num_return_sequences = 1):
    """following original needlebench evaluation"""
    keywords = answer['keywords']
    reference = answer['reference']
    alpha = 0.2
    # generate prediction
    start = time.time()
    encoded_prompt = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded_prompt.input_ids
    input_ids = input_ids.to(model.device)
    attention_mask = encoded_prompt.attention_mask.to(model.device)
    len_token = input_ids.shape[-1]
    # print("len tokens", len_token)
    answer_len = len(tokenizer(reference).input_ids) - 1 # drop BOS
    print('input_ids.shape: ', input_ids.shape)
    try:
        generation_output = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=answer_len+20, \
            num_return_sequences=num_return_sequences, do_sample=True, temperature=0.7
    )
    except:
        return None, {'input_tokens': len_token, 'output_tokens': -1, 'avg_time': -1}
    span = time.time() - start
    output_tokens, predictions = [], []
    for i in range(num_return_sequences):
        output_token = generation_output[i][len_token:]
        output_tokens.append(output_token)
        predictions.append(tokenizer.decode(output_token))

    return predictions, {'input_tokens': len_token, 'output_tokens': [len(o) for o in output_tokens], 'avg_time': round(span/num_return_sequences,3)}

# PASS
def evaluate_niah_with_scorer(prompt, answer, tokenizer, model):
    """following original needlebench evaluation"""
    keywords = answer['keywords']
    reference = answer['reference']
    alpha = 0.2
    # generate prediction
    start = time.time()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(model.device)
    len_token = input_ids.shape[-1]
    # print("len tokens", len_token)
    # TODO: config tokenizer
    answer_len = len(tokenizer(reference).input_ids) - 1 # drop BOS
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=answer_len+10, num_beams=1
    )
    span = time.time() - start
    prediction = tokenizer.decode(generation_output[0][len_token:])
    # prediction = prediction.split('\n')[0]
    # prediction = prediction.split('.')[0]
    # print("[prediction]:  ", prediction)
    # print("[ground truth]: ", reference)
    # print("--------")

    prediction_words = prediction.split()
    reference_words = reference.split()
    # if keyword exists
    if any([kw in prediction_words for kw in keywords]):
        score = 1
    else:
        dist = levenshtein_distance_word_level(prediction, reference)
        score = alpha * (1 - dist / max(len(prediction_words), len(reference_words)))
    return score, {'input_tokens': len_token, 'generation_time': span}
