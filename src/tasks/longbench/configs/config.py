# longbench
import json
from src.tasks.longbench.metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)
DATA2MAXLEN = json.load(open("src/tasks/longbench/configs/dataset2maxlen.json", "r"))
DATA2PROMPT = json.load(open("src/tasks/longbench/configs/dataset2prompt.json", "r"))
DATA2PROMPT_CQ = json.load(open("src/tasks/longbench/configs/dataset2prompt_cq.json", "r"))
DATA2SPLIT = {'trec': 'Question', 'triviaqa': 'Passage', 'samsum': 'Dialogue', 'passage_count': 'Paragraph', 'passage_retrieval_en': 'Paragraph'}
DATA2METRIC = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}