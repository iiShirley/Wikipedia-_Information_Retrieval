"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from collections import Counter, defaultdict
import numpy as np
from indexing import InvertedIndex


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """
    # TODO: This class is responsible for returning a list of sorted relevant documents.
    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
            raw_text_dict: A dictionary mapping a document ID to the raw string of the document (Not needed for HW1)
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer #choose the scorer method
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        """
        # 1. Tokenize query (Hint: Also apply stopwords filtering to the tokenized query)

        # 2.1 For each token in the tokenized query, find out all documents that contain it and counting its frequency within each document.
        # Hint 1: To understand why we need the info above, pay attention to docid and doc_word_counts, 
        #    located in the score() function within the RelevanceScorer class
        # Hint 2: defaultdict(Counter) works well in this case, where we store {docids : {query_tokens : counts}}, 
        #         or you may choose other approaches

        # 2.2 Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes) 
        #        for each relevant document determined in 2.1

        # 3. Return **sorted** results as format [{docid: 100, score:0.5}, {{docid: 10, score:0.2}}]


        # 1. Tokenize query
        query_tokens = self.tokenize(query)
        if self.stopwords:
            # Replace stopwords with None instead of removing them to preserve query length
            query_tokens = [tok if tok not in self.stopwords else None for tok in query_tokens]

        # 2. get word frequency
        query_word_counts = Counter(query_tokens)

        # 3. find all documents with the terms
        doc_word_counts = defaultdict(Counter)  # {docid: {term: count}}

        for term in query_word_counts.keys():
            postings = self.index.get_postings(term)
            for posting in postings:
                if len(posting) == 2:
                    # BasicInvertedIndex: (docid, count)
                    docid, count = posting
                elif len(posting) == 3:
                    # PositionalInvertedIndex: (docid, freq, positions)
                    docid, count, positions = posting
                else:
                    continue
                doc_word_counts[docid][term] = count

        # 4. calculate the score of each document
        results = []
        for docid, d_wc in doc_word_counts.items():
            score = self.scorer.score(docid, d_wc, query_word_counts)
            results.append((docid, score))

        # 5. rankin
        results.sort(key=lambda x: x[1], reverse=True)

        return results


class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # NOTE: Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document that are also found in the query, 
                                and their frequencies within the document.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO: Implement unnormalized cosine similarity on word count vectors (inner product)
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        score = 0.0
        for term, q_count in query_word_counts.items():
            d_count = doc_word_counts.get(term, 0)
            score += d_count * q_count  # dot product
        return score


# TODO: Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.mu = parameters.get('mu', 2000)

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        dl = self.index.get_doc_metadata(docid).get("length", 1)
        collection_length = self.index.statistics['stored_total_token_count']

        score = 0.0
        # Part 1: loop over query terms
        for term, qf in query_word_counts.items():
            cd = doc_word_counts.get(term, 0)
            cf = self.index.statistics['vocab'].get(term, 0)  # collection frequency
            p_w_C = cf / collection_length if collection_length > 0 else 0.0

            if p_w_C > 0:  # avoid div0
                # Formula: log(1 + (c(w,d) / (μ ⋅ p(w|C))))
                score += qf * np.log(1 + (cd / (self.mu * p_w_C)))

        # Part 2: normalized document length term (outside query loop)
        # Formula: |q| ⋅ log(μ / (|d| + μ))
        query_length = sum(query_word_counts.values())  # Total query term frequency
        score += query_length * np.log(self.mu / (dl + self.mu))

        return score


# TODO: Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters.get('b', 0.75)
        self.k1 = parameters.get('k1', 1.2)
        self.k3 = parameters.get('k3', 8)

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score
        N = self.index.statistics['number_of_documents']
        avdl = self.index.statistics['mean_document_length']
        dl = self.index.get_doc_metadata(docid).get("length")

        score = 0.0
        for term, qf in query_word_counts.items():
            cd = doc_word_counts.get(term, 0)
            if cd == 0:
                continue
            df = self.index.get_term_metadata(term).get("doc_frequency", 1)

            idf = np.log((N - df + 0.5) / (df + 0.5))
            tf = ((self.k1 + 1) * cd) / (self.k1 * ((1 - self.b) + self.b * dl / avdl) + cd)
            qtf = ((self.k3 + 1) * qf) / (self.k3 + qf)

            score += idf * tf * qtf
        return score


# TODO: Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters.get('b', 0.2)

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        N = self.index.statistics['number_of_documents']
        avdl = self.index.statistics['mean_document_length']
        dl = self.index.get_doc_metadata(docid).get("length", 1)

        score = 0.0
        for term, qf in query_word_counts.items():
            cd = doc_word_counts.get(term, 0)
            if cd == 0:
                continue

            tf = 1 + np.log(1 + np.log(cd))
            norm = (1 - self.b) + self.b * (dl / avdl)
            df = self.index.get_term_metadata(term).get("doc_frequency", 1)
            idf = np.log((N + 1) / df)

            score += qf * (tf / norm) * idf
        return score



# TODO: Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return the score
        score = 0.0
        N = self.index.statistics['number_of_documents']
        for term, q_count in query_word_counts.items():
            cd = doc_word_counts.get(term, 0)
            if cd == 0:
                continue
            tf = np.log(cd + 1)   # log(cd(wi)+1)
            df = self.index.get_term_metadata(term).get("doc_frequency", 1)
            idf = np.log(N / df) + 1   # log(|D|/df(wi)) + 1
            score += tf * idf
        return score

# TODO: Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # penalize long dolcument based on TF-IDF
        score = 0.0
        N = self.index.statistics['number_of_documents']
        dl = self.index.get_doc_metadata(docid).get("length", 1)
        for term, q_count in query_word_counts.items():
            tf = np.log1p(doc_word_counts.get(term, 0))
            df = self.index.get_term_metadata(term).get("doc_frequency", 1)
            idf = np.log((N / df) + 1)
            score += tf * idf * q_count
        return score / (1 + 0.0005 * dl)
