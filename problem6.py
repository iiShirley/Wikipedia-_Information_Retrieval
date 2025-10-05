import time
import tracemalloc
# import matplotlib.pyplot as plt
from indexing import Indexer, IndexType
from document_preprocessor import SplitTokenizer, RegexTokenizer


def benchmark_indexing(index_type, dataset_path, doc_counts):
    """
    Benchmark indexing time and memory usage.
    """
    times = []
    memories = []

    for n_docs in doc_counts:
        print(f"\n>>> Building {index_type.value} with {n_docs} docs")

        # start measuring memory
        tracemalloc.start()
        start_time = time.time()

        # build index - use SplitTokenizer for better performance
        index = Indexer.create_index(
            index_type=index_type,
            dataset_path=dataset_path,
            document_preprocessor=SplitTokenizer(),  # Faster than RegexTokenizer
            stopwords=None,
            minimum_word_frequency=0,
            text_key="text",
            max_docs=n_docs
        )

        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # record
        times.append(elapsed)
        memories.append(peak / (1024 * 1024))  # convert to MB

        print(f"Docs: {n_docs}, Time: {elapsed:.2f}s, Peak Memory: {peak/1024/1024:.2f} MB")

    return times, memories


if __name__ == "__main__":
    dataset_path = "../../data/wikipedia_200k_dataset.jsonl.gz"
    doc_counts = [10, 100, 1000, 10000]

    # Benchmark BasicInvertedIndex
    basic_times, basic_mem = benchmark_indexing(IndexType.BasicInvertedIndex, dataset_path, doc_counts)

    # Benchmark PositionalIndex
    pos_times, pos_mem = benchmark_indexing(IndexType.PositionalIndex, dataset_path, doc_counts)

    # ---- Plot Time ----
    plt.figure()
    plt.plot(doc_counts, basic_times, marker="o", label="BasicInvertedIndex")
    plt.plot(doc_counts, pos_times, marker="o", label="PositionalIndex")
    plt.xlabel("Number of Documents")
    plt.ylabel("Time (s)")
    plt.title("Indexing Time Comparison")
    plt.legend()
    plt.savefig("problem6_time.png")

    # ---- Plot Memory ----
    plt.figure()
    plt.plot(doc_counts, basic_mem, marker="o", label="BasicInvertedIndex")
    plt.plot(doc_counts, pos_mem, marker="o", label="PositionalIndex")
    plt.xlabel("Number of Documents")
    plt.ylabel("Memory (MB)")
    plt.title("Indexing Memory Comparison")
    plt.legend()
    plt.savefig("problem6_memory.png")

    # ---- Save Results ----
    with open("problem6_results.txt", "w") as f:
        f.write("Docs\tBasicTime(s)\tBasicMem(MB)\tPosTime(s)\tPosMem(MB)\n")
        for i, n in enumerate(doc_counts):
            f.write(f"{n}\t{basic_times[i]:.2f}\t{basic_mem[i]:.2f}\t{pos_times[i]:.2f}\t{pos_mem[i]:.2f}\n")

    print("\nDone! Results saved to problem6_results.txt, problem6_time.png, problem6_memory.png")
