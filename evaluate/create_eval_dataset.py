from datasets import Dataset
from rag import graph_with_retriever

# Sample questions about the 2025 stock market crash
questions = [
    "What caused the 2025 stock market crash?",
    "What was the impact of Trump's tariffs on the Dow Jones on April 3, 2025?",
    "How did China respond to the U.S. tariffs announced on April 2, 2025?",
    "What was the percentage drop in the Nikkei 225 on April 4, 2025?",
    "What did Trump say about the stock market crash during his golf trip?",
]

# Run queries through the RAG pipeline
eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

# Ground-truth answers (manually curated from 2025_crash.txt)
ground_truths = [
    "The 2025 stock market crash was caused by sweeping tariffs and trade wars initiated by U.S. President Donald Trump, announced on April 2, 2025, as 'Liberation Day'. These policies heightened tensions with allies and led to global economic uncertainty.",
    "On April 3, 2025, the Dow Jones index lost 1,679.39 points, a decline of 3.98%.",
    "China announced retaliatory tariffs of 34% against the United States on April 2, 2025, starting on April 10, in response to the U.S. tariffs.",
    "The Nikkei 225 experienced a 2.75% loss on April 4, 2025, closing at 33,780.58.",
    "During his golf trip on April 7, 2025, Trump responded to criticism by posting on social media, telling people 'Don’t be a PANICAN (a new party based on Weak and Stupid people!)'.",
]

for question, gt in zip(questions, ground_truths):
    # Run the query through the RAG pipeline
    result = graph_with_retriever.invoke(
        {"query": question, "context": [], "answer": ""}
    )

    eval_data["question"].append(question)
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append(result["context"])
    eval_data["ground_truth"].append(gt)

# Save as a HuggingFace Dataset
dataset = Dataset.from_dict(eval_data)
dataset.save_to_disk("eval_dataset")
