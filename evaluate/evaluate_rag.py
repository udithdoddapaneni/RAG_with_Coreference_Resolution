
import pandas as pd
import sys
from ragas.metrics import RougeScore, BleuScore
from ragas.dataset_schema import SingleTurnSample
from bert_score import BERTScorer
sys.path.append(".")

ROUGE = RougeScore()
BLEU = BleuScore()
BERT = BERTScorer(model_type="bert-base-uncased")

dataset = pd.read_csv("eval_dataset.csv")

results = {"rouge-L":0, "bleu":0, "bert-precision":0, "bert-recall":0, "bert-f1":0}
r, c = dataset.shape
P, R, F1 = BERT.score(dataset["answer"].to_list(), dataset["ground_truth"].to_list())
for i, row in dataset.iterrows():
    sample = SingleTurnSample(response=row["answer"], reference=row["ground_truth"])
    rouge = ROUGE.single_turn_score(sample)
    bleu = BLEU.single_turn_score(sample)
    results["rouge-L"] += rouge/r
    results["bleu"] += bleu/r
results["bert-f1"] = F1.mean().item()
results["bert-recall"] = R.mean().item()
results["bert-precision"] = P.mean().item()
print(results)
