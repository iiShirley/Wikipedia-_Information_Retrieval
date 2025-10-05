"""
NOTE: We've curated a set of query-document relevance scores for you to use in this part of the assignment. 
You can find 'relevance.test.csv', where the 'rel' column contains scores of the following relevance levels: 
1 (non-relevant) to 5 (very relevant). When you calculate MAP, treat 4s and 5s as relevant documents. 
Treat search results from your ranking function that are not listed in the file as non-relevant.
"""
import math
import csv
from tqdm import tqdm
import numpy as np


def map_score(search_result_relevances: list[int], cut_off: int = 10) -> float:
    """
    Calculates the mean average precision score given a list of labeled search results, where
    each item in the list corresponds to a document that was retrieved and is rated as 0 or 1
    for whether it was relevant.

    Args:
        search_result_relevances: A list of 0/1 values for whether each search result returned by your
            ranking function is relevant
        cut_off: The search result rank to stop calculating MAP.
            The default cut-off is 10; calculate MAP@10 to score your ranking function.

    Returns:
        The MAP score
    """
    relevant_count = 0
    precision_sum = 0.0
    total_relevant = sum(search_result_relevances[:cut_off])

    if total_relevant == 0: # for MAp, when relevance score is 0, we ignore that document
        return 0.0

    for k, rel in enumerate(search_result_relevances[:cut_off], start=1):
        if rel == 1:
            relevant_count += 1
            precision_sum += relevant_count / k

    return precision_sum / total_relevant


def ndcg_score(search_result_relevances: list[float],
               ideal_relevance_score_ordering: list[float], cut_off: int = 10):
    """
    Calculates the normalized discounted cumulative gain (NDCG) given a lists of relevance scores.
    Relevance scores can be ints or floats, depending on how the data was labeled for relevance.

    Args:
        search_result_relevances: A list of relevance scores for the results returned by your ranking function
            in the order in which they were returned
            These are the human-derived document relevance scores, *not* the model generated scores.
        ideal_relevance_score_ordering: The list of relevance scores for results for a query, sorted by relevance score
            in descending order
            Use this list to calculate IDCG (Ideal DCG).

        cut_off: The default cut-off is 10.

    Returns:
        The NDCG score
    """
    def dcg(rels):
        return sum((2**rel - 1) / math.log2(i+2) for i, rel in enumerate(rels[:cut_off]))

    dcg_val = dcg(search_result_relevances)
    idcg_val = dcg(ideal_relevance_score_ordering)

    if idcg_val == 0:
        return 0.0
    return dcg_val / idcg_val


def run_relevance_tests(relevance_data_filename: str, ranker) -> dict[str, float]:
    # TODO: Implement running relevance test for the search system for multiple queries.
    """
    Measures the performance of the IR system using metrics, such as MAP and NDCG.

    Args:
        relevance_data_filename: The filename containing the relevance data to be loaded
        ranker: A ranker configured with a particular scoring function to search through the document collection.
            This is probably either a Ranker or a L2RRanker object, but something that has a query() method.

    Returns:
        A dictionary containing both MAP and NDCG scores
    """
    # TODO: Load the relevance dataset

    # TODO: Run each of the dataset's queries through your ranking function

    # TODO: For each query's result, calculate the MAP and NDCG for every single query and average them out

    # NOTE: MAP requires using binary judgments of relevant (1) or not (0). You should use relevance
    #       scores of (1,2,3) as not-relevant, and (4,5) as relevant.

    # NOTE: Treat search results from your ranking function that are not listed in the relevance_data_filename as non-relevant

    # NOTE: NDCG can use any scoring range, so no conversion is needed.

    # TODO: Compute the average MAP and NDCG across all queries and return the scores
    # NOTE: You should also return the MAP and NDCG scores for each query in a list
    return {'map': 0, 'ndcg': 0, 'map_list': [], 'ndcg_list': []}


if __name__ == '__main__':
    pass
