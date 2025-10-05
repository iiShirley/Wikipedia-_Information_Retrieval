import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from indexing import Indexer, IndexType
from document_preprocessor import SplitTokenizer
from ranker import (
    Ranker, WordCountCosineSimilarity, DirichletLM, TF_IDF,
    BM25, PivotedNormalization, YourRanker
)
from relevance import map_score, ndcg_score


def run_relevance_tests(relevance_data_filename: str, ranker, cutoff: int = 10):
    """
    Run MAP and NDCG tests for all queries in the dataset.
    """
    relevance_data = {}
    with open(relevance_data_filename, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row['query']
            docid = int(row['docid'])
            rel = int(row['rel'])
            if query not in relevance_data:
                relevance_data[query] = {}
            relevance_data[query][docid] = rel

    map_scores, ndcg_scores = [], []

    for query_text, doc_rels in tqdm(relevance_data.items(), desc="Processing queries", unit="query"):
        results = ranker.query(query_text)
        result_docids = [docid for docid, _ in results[:cutoff]]

        # MAP needs binary relevance
        binary_rels = [1 if docid in doc_rels and doc_rels[docid] >= 4 else 0
                       for docid in result_docids]

        # NDCG uses graded relevance
        graded_rels = [doc_rels.get(docid, 0) for docid in result_docids]
        ideal_order = sorted(doc_rels.values(), reverse=True)

        map_scores.append(map_score(binary_rels, cutoff))
        ndcg_scores.append(ndcg_score(graded_rels, ideal_order, cutoff))

    return {
        'map': sum(map_scores) / len(map_scores),
        'ndcg': sum(ndcg_scores) / len(ndcg_scores),
        'map_list': map_scores,
        'ndcg_list': ndcg_scores
    }


def problem10_main(dataset_path, relevance_file):
    """
    Main entry for Problem 10.
    Builds the index, evaluates rankers, saves table + plots.
    
    Tokenizer Choice: SplitTokenizer
    - Reason: Faster than RegexTokenizer for large datasets
    - Parameters: Default settings (lowercase=True, no multiword expressions)
    - Document Count: 30,000 documents (balanced for performance and quality)
    - Expected performance: ~15-25 minutes total execution time
    """
    tokenizer = SplitTokenizer()
    stopwords = set()

    # build index with Indexer
    print("🔨 Building index... (this may take a while for large datasets)")
    # Use 30,000 documents for good balance between performance and results quality
    max_docs = 30000
    
    index = Indexer.create_index(
        IndexType.BasicInvertedIndex,
        dataset_path,
        tokenizer,
        stopwords,
        minimum_word_frequency=0,
        text_key="text",
        max_docs=max_docs
    )
    print(f"✅ Index built successfully. Total documents: {index.statistics['number_of_documents']}")

    scorers = {
        "WordCountCosine": WordCountCosineSimilarity(index),
        "DirichletLM": DirichletLM(index),
        "TF_IDF": TF_IDF(index),
        "BM25": BM25(index),
        "PivotedNorm": PivotedNormalization(index),
        "YourRanker": YourRanker(index),
    }

    results = {}
    print("tarting ranker evaluation...")
    for name, scorer in tqdm(scorers.items(), desc="Evaluating rankers", unit="ranker"):
        print(f"Evaluating {name}...")
        ranker = Ranker(index, tokenizer, stopwords, scorer)
        scores = run_relevance_tests(relevance_file, ranker)
        results[name] = scores
        print(f"✅ {name} completed. MAP: {scores['map']:.4f}, NDCG: {scores['ndcg']:.4f}")

    # Save summary table
    print("📊 Generating results table...")
    df = pd.DataFrame({
        name: [scores['map'], scores['ndcg']]
        for name, scores in results.items()
    }, index=["MAP", "NDCG"])
    print("Results Summary:")
    print(df)
    df.to_csv("problem10_results.csv")
    print("💾 Results saved to problem10_results.csv")

    # Violin plots
    print("🎨 Generating violin plots...")
    for metric in tqdm(["map_list", "ndcg_list"], desc="Creating plots", unit="plot"):
        plot_df = pd.DataFrame({
            name: scores[metric] for name, scores in results.items()
        })
        plot_df = plot_df.melt(var_name="Ranker", value_name=metric)
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=plot_df, x="Ranker", y=metric, inner="point")
        plt.title(f"Distribution of {metric.upper()} across queries")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(f"problem10_{metric}.png")
        plt.close()
        print(f"📊 Saved plot: problem10_{metric}.png")
    
    print("🎉 Problem 10 completed successfully!")


if __name__ == "__main__":
    dataset_path = "/home/shirleyi/HW1/data/wikipedia_200k_dataset.jsonl.gz"
    relevance_file = "/home/shirleyi/HW1/data/relevance.test.csv"
    problem10_main(dataset_path, relevance_file)
