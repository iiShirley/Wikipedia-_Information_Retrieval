import time
import gzip
import json
import matplotlib.pyplot as plt
from document_preprocessor import SplitTokenizer, RegexTokenizer, SpaCyTokenizer

# =====================
# Step 1. Load the first 1000 documents
# =====================
def load_documents(path, limit=1000):
    docs = []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            doc = json.loads(line)
            # ⚠️ Check the key name: could be "text" or "body"
            docs.append(doc["text"])
    return docs
docs = load_documents("../../data/wikipedia_200k_dataset.jsonl.gz", limit=1000)





# =====================
# Step 2. Define tokenizers
# =====================
tokenizers = {
    "SplitTokenizer": SplitTokenizer(),
    "RegexTokenizer": RegexTokenizer(),
    "SpaCyTokenizer": SpaCyTokenizer()
}

results = {}

# =====================
# Step 3. Run experiments
# =====================
for name, tok in tokenizers.items():
    print(f"Running {name} ...")
    start = time.time()
    for d in docs:
        tokens = tok.tokenize(d)
    end = time.time()
    total_time = end - start
    avg_time = total_time / len(docs)
    results[name] = (total_time, avg_time)
    print(f"{name}: Total={total_time:.2f}s, Avg={avg_time:.4f}s/doc")

# =====================
# Step 4. Save results to a text file
# =====================
with open("problem3_results.txt", "w") as f:
    f.write("Tokenizer\tTotal_Time(s)\tAvg_Time(s/doc)\n")
    for name, (total, avg) in results.items():
        f.write(f"{name}\t{total:.2f}\t{avg:.4f}\n")

# =====================
# Step 5. Plot a bar chart for comparison
# =====================
plt.bar(results.keys(), [v[0] for v in results.values()])
plt.ylabel("Time for 1000 docs (seconds)")
plt.title("Tokenizer Efficiency Comparison")
plt.savefig("problem3_time.png")

print("Done! Results saved to problem3_results.txt and problem3_time.png")
