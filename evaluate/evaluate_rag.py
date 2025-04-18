import os

from datasets import load_from_disk
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

# Set OpenAI API key (or use an open-source LLM)
os.environ["OPENAI_API_KEY"] = (
    "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"
)

# Load the evaluation dataset
dataset = load_from_disk("eval_dataset")

# Define the LLM for evaluation (e.g., GPT-4o)
evaluator_llm = ChatOpenAI(model="gpt-4o")

# Metrics to evaluate
metrics = [
    faithfulness,  # Checks if the answer is consistent with the context
    answer_relevancy,  # Evaluates if the answer addresses the query
    context_precision,  # Assesses if relevant context is ranked higher
    context_recall,  # Checks if all relevant ground-truth context is retrieved
]

# Run RAGAS evaluation
results = evaluate(dataset=dataset, metrics=metrics, llm=evaluator_llm)

# Print results
print(results)
