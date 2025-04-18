from datasets import Dataset
import sys
import pandas as pd

sys.path.append(".")
from RAG.rag import graph_with_retriever, graph_direct, UpdateDatabase

UpdateDatabase(resolve_refs=True, only_pronouns=True)

dataset_csv = pd.read_csv("evaluate/questions_groundtruths.csv")
questions = dataset_csv["questions"].to_list()
# Run queries through the RAG pipeline
eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

ground_truths = dataset_csv["groundtruths"].to_list()

for question, gt in zip(questions, ground_truths):
    # Run the query through the RAG pipeline
    result = graph_with_retriever.invoke(
        {"query": question, "context": [], "answer": ""}
    )

    eval_data["question"].append(question)
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append(result["context"])
    eval_data["ground_truth"].append(gt)

# Save as a CSV
dataset = Dataset.from_dict(eval_data)
dataset.to_csv("eval_dataset.csv")