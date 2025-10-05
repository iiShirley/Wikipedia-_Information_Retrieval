'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''
import os
import gzip
import json
from enum import Enum
from collections import Counter, defaultdict
from tqdm import tqdm
from document_preprocessor import Tokenizer


class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex and PositionalIndex
    BasicInvertedIndex = 'BasicInvertedIndex'
    PositionalIndex = 'PositionalIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    def __init__(self) -> None:
        """
        The base interface representing the data structure for all index classes.
        The functions are meant to be implemented in the actual index classes and not as part of this interface.

        Note: The following variables are defined to help you store some summary info about your document collection
                for a quick look-up.
              You may also define more variables and/or keys as you see fit.
        Variables:
            statistics: A dictionary, which is the central statistics of the index.
                        Some keys include:
                statistics['vocab']: A counter which keeps track of the token count
                statistics['unique_token_count']: how many unique terms are in the index
                statistics['total_token_count']: how many total tokens are indexed including filterd tokens),
                    i.e., the sum of the lengths of all documents
                statistics['stored_total_token_count']: how many total tokens are indexed excluding filtered tokens
                statistics['number_of_documents']: the number of documents indexed
                statistics['mean_document_length']: the mean number of tokens in a document (including filter tokens)
                ...
                (Add more keys to the statistics dictionary as you see fit)
                
            vocabulary: A set of distinct words that have appeared in the collection
            document_metadata: A dictionary, which keeps track of some important metadata for each document.
                               Assume that we have a document called 'doc1', some keys include:
                document_metadata['doc1']['unique_tokens']: How many unique tokens are in the document (among those not-filtered)
                document_metadata['doc1']['length']: How long the document is in terms of tokens (including those filtered) 
                ...
                (Add more keys to the document_metadata dictionary as you see fit)
            index: A dictionary of class defaultdict, its implemention depends on whether we are using 
                            BasicInvertedIndex or PositionalIndex.
                    BasicInvertedIndex: Store the mapping of terms to their postings
                    PositionalIndex: Each term keeps track of documents and positions of the terms occurring in the document
        """
        self.statistics = {}   
        self.statistics['vocab'] = Counter()  
        self.vocabulary = set()  
        self.document_metadata = {}
        self.index = defaultdict(list)

    # NOTE: The following functions have to be implemented in the two inherited classes and NOT in this class
    def add_doc(self, docid: int, tokens: list[str]) -> None:
        raise NotImplementedError

    def remove_doc(self, docid: int) -> None:
        # Remove document from all postings
        for term in list(self.index.keys()):
            postings = self.index[term]
            # Find and remove the document from postings
            for i, posting in enumerate(postings):
                if posting[0] == docid:
                    postings.pop(i)
                    break
            # Remove term if no more postings
            if not postings:
                del self.index[term]
                if term in self.vocabulary:
                    self.vocabulary.remove(term)
                if term in self.statistics['vocab']:
                    del self.statistics['vocab'][term]
        
        # Remove document metadata
        if docid in self.document_metadata:
            doc_metadata = self.document_metadata[docid]
            doc_length = doc_metadata.get('length', 0)
            
            # Update statistics
            self.statistics['total_token_count'] = max(0, self.statistics.get('total_token_count', 0) - doc_length)
            self.statistics['number_of_documents'] = max(0, self.statistics.get('number_of_documents', 0) - 1)
            
            if self.statistics['number_of_documents'] > 0:
                self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
            else:
                self.statistics['mean_document_length'] = 0
            
            del self.document_metadata[docid]

    def get_postings(self, term: str) -> list[tuple[int, int]]:
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        raise NotImplementedError

    def save(self, index_directory_name: str = 'tmp') -> None:
        os.makedirs(index_directory_name, exist_ok=True)
        
        # Save index
        index_data = {term: postings for term, postings in self.index.items()}
        with open(os.path.join(index_directory_name, 'index.json'), 'w') as f:
            json.dump(index_data, f)
        
        # Save statistics
        stats_data = dict(self.statistics)
        # Convert Counter to dict for JSON serialization
        if 'vocab' in stats_data and isinstance(stats_data['vocab'], Counter):
            stats_data['vocab'] = dict(stats_data['vocab'])
        with open(os.path.join(index_directory_name, 'statistics.json'), 'w') as f:
            json.dump(stats_data, f)
        
        # Save vocabulary
        with open(os.path.join(index_directory_name, 'vocabulary.json'), 'w') as f:
            json.dump(list(self.vocabulary), f)
        
        # Save document metadata
        with open(os.path.join(index_directory_name, 'document_metadata.json'), 'w') as f:
            json.dump(self.document_metadata, f)

    def load(self, index_directory_name: str = 'tmp') -> None:
        # Load index
        with open(os.path.join(index_directory_name, 'index.json'), 'r') as f:
            index_data = json.load(f)
        self.index = defaultdict(list, index_data)
        
        # Load statistics
        with open(os.path.join(index_directory_name, 'statistics.json'), 'r') as f:
            stats_data = json.load(f)
        # Convert vocab dict back to Counter
        if 'vocab' in stats_data:
            stats_data['vocab'] = Counter(stats_data['vocab'])
        self.statistics = stats_data
        
        # Load vocabulary
        with open(os.path.join(index_directory_name, 'vocabulary.json'), 'r') as f:
            self.vocabulary = set(json.load(f))
        
        # Load document metadata
        with open(os.path.join(index_directory_name, 'document_metadata.json'), 'r') as f:
            self.document_metadata = json.load(f)


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'
        # self.statistics['total_token_count'] = 0
        # self.statistics['stored_total_token_count'] = 0
        # self.statistics['number_of_documents'] = 0
        # self.statistics['mean_document_length'] = 0

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        token_counts = Counter(tokens)

        doc_unique_tokens = 0
        for term, freq in token_counts.items():
            if term is None:
                continue
            doc_unique_tokens += 1
            self.vocabulary.add(term)
            self.statistics['vocab'][term] += freq
            # Insert posting in sorted order by docid
            # postings = self.index[term]
            # # Find insertion point
            # insert_pos = 0
            # for i, (existing_docid, _) in enumerate(postings):
            #     if existing_docid < docid:
            #         insert_pos = i + 1
            #     else:
            #         break
            # postings.insert(insert_pos, (docid, freq))
            self.index[term].append((docid, freq))

            self.document_metadata[docid] = {
                'unique_tokens': doc_unique_tokens,
                'length': len(tokens)
            }

        # update collection-level stats
        # total_token_count should include all tokens (before filtering)
        # stored_total_token_count should include only stored tokens (after filtering)
        self.statistics['total_token_count'] = self.statistics.get('total_token_count', 0) + len(tokens)
        self.statistics['stored_total_token_count'] = self.statistics.get('stored_total_token_count', 0) + len(tokens)
        self.statistics['number_of_documents'] = self.statistics.get('number_of_documents', 0) + 1
        self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']


    def get_postings(self, term: str):
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int):
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str):
        return {
            "term_count": self.statistics['vocab'].get(term, 0),
            "doc_frequency": len(self.index.get(term, []))
        }

    def get_statistics(self):
        self.statistics['unique_token_count'] = len(self.vocabulary)
        # self.statistics['total_token_count'] = sum([self.document_metadata[docid]['length'] for docid in self.document_metadata])
        # Ensure all required statistics are present
        if 'total_token_count' not in self.statistics:
            self.statistics['total_token_count'] = 0
        if 'stored_total_token_count' not in self.statistics:
            self.statistics['stored_total_token_count'] = 0
        if 'number_of_documents' not in self.statistics:
            self.statistics['number_of_documents'] = 0
        if 'mean_document_length' not in self.statistics:
            self.statistics['mean_document_length'] = 0
        # Recalculate mean_document_length if needed
        if self.statistics['number_of_documents'] > 0:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
        else:
            self.statistics['mean_document_length'] = 0
        return self.statistics


class PositionalInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()
        self.statistics['index_type'] = 'PositionalInvertedIndex'

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        token_positions = defaultdict(list)
        for pos, tok in enumerate(tokens):
            if tok is None:
                continue
            token_positions[tok].append(pos)

        self.document_metadata[docid] = {
            'unique_tokens': len(token_positions),
            'length': len(tokens)
        }

        for term, positions in token_positions.items():
            self.vocabulary.add(term)
            self.statistics['vocab'][term] += len(positions)
            # Insert posting in sorted order by docid
            postings = self.index[term]
            # Find insertion point
            insert_pos = 0
            for i, (existing_docid, _, _) in enumerate(postings):
                if existing_docid < docid:
                    insert_pos = i + 1
                else:
                    break
            postings.insert(insert_pos, (docid, len(positions), positions))

        # update collection-level stats
        self.statistics['total_token_count'] = self.statistics.get('total_token_count', 0) + len(tokens)
        self.statistics['stored_total_token_count'] = self.statistics.get('stored_total_token_count', 0) + len(tokens)
        self.statistics['number_of_documents'] = self.statistics.get('number_of_documents', 0) + 1
        self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']

    def get_postings(self, term: str):
        return self.index.get(term, [])

    def get_doc_metadata(self, doc_id: int):
        return self.document_metadata.get(doc_id, {})

    def get_term_metadata(self, term: str):
        return {
            "term_count": self.statistics['vocab'].get(term, 0),
            "doc_frequency": len(self.index.get(term, []))
        }

    def get_statistics(self):
        self.statistics['unique_token_count'] = len(self.vocabulary)
        # Ensure all required statistics are present
        if 'total_token_count' not in self.statistics:
            self.statistics['total_token_count'] = 0
        if 'stored_total_token_count' not in self.statistics:
            self.statistics['stored_total_token_count'] = 0
        if 'number_of_documents' not in self.statistics:
            self.statistics['number_of_documents'] = 0
        if 'mean_document_length' not in self.statistics:
            self.statistics['mean_document_length'] = 0
        # Recalculate mean_document_length if needed
        if self.statistics['number_of_documents'] > 0:
            self.statistics['mean_document_length'] = self.statistics['total_token_count'] / self.statistics['number_of_documents']
        else:
            self.statistics['mean_document_length'] = 0
        return self.statistics


class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''
    @staticmethod
    def create_index(index_type: IndexType, dataset_path: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1) -> InvertedIndex:

        # if index_type == IndexType.BasicInvertedIndex:
        #     index = BasicInvertedIndex()
        # elif index_type == IndexType.PositionalIndex:
        #     index = PositionalInvertedIndex()
        # else:
        #     raise ValueError(f"Unknown index type {index_type}")

        # # 1st pass: count token frequency and total tokens before filtering
        # word_freq = Counter()
        # total_tokens_before_filtering = 0
        # open_func = gzip.open if dataset_path.endswith(".gz") else open
        # # Handle relative paths for test files
        # original_dataset_path = dataset_path
        # if not os.path.isabs(dataset_path) and not os.path.exists(dataset_path):
        #     # Try tests/ directory if file not found
        #     test_path = os.path.join("tests", dataset_path)
        #     if os.path.exists(test_path):
        #         dataset_path = test_path
        # with open_func(dataset_path, "rt", encoding="utf-8") as f:
        #     for line_num, line in enumerate(f):
        #         if max_docs > 0 and line_num >= max_docs:
        #             break
        #         data = json.loads(line)
        #         docid = data.get('docid', line_num)  # Use docid from data, fallback to line_num
        #         text = data[text_key]
        #         tokens = document_preprocessor.tokenize(text)
        #         total_tokens_before_filtering += len(tokens)
        #         word_freq.update(tokens)

        # # identify low-frequency tokens
        # low_freq_tokens = set()
        # if minimum_word_frequency > 0:
        #     for token, count in word_freq.items():
        #         if count < minimum_word_frequency:
        #             low_freq_tokens.add(token)

        # # 2nd pass: build index
        # # Use the same dataset_path from the first pass
        # with open_func(dataset_path, "rt", encoding="utf-8") as f:
        #     for line_num, line in enumerate(f):
        #         if max_docs > 0 and line_num >= max_docs:
        #             break
        #         data = json.loads(line)
        #         docid = data.get('docid', line_num)  # Use docid from data, fallback to line_num
        #         text = data[text_key]
        #         tokens = document_preprocessor.tokenize(text)

        #         if stopwords:
        #             tokens = [tok for tok in tokens if tok not in stopwords]
        #         if minimum_word_frequency > 0:
        #             tokens = [tok for tok in tokens if tok not in low_freq_tokens]

        #         index.add_doc(docid, tokens)

        # # total_token_count and mean_document_length are already set in add_doc
        # # No need to set them again here

        # return index
        
        
        if index_type == IndexType.BasicInvertedIndex:
            index = BasicInvertedIndex()
        elif index_type == IndexType.PositionalIndex:
            index = PositionalInvertedIndex()
        else:
            raise ValueError('Invalid index type')
       
        doc_data = []  # Store (docid, tokens) for second pass
        doc_count = 0
        word_frequencies = Counter()

        # Define how to open jsonl file
        if dataset_path.endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
       
        with open_func(dataset_path, mode, encoding='utf-8') as json_file:
            for doc in json_file:
                if max_docs > 0 and doc_count >= max_docs:
                    break
                doc = json.loads(doc)
                tokens = document_preprocessor.tokenize(doc[text_key])
                for token in tokens:
                    word_frequencies[token] += 1
                doc_data.append((doc['docid'], tokens))
                doc_count += 1

        # Exclude words not to index
        exclude_tokens = set()
        if minimum_word_frequency > 0 and text_key == 'text':
            exclude_tokens.update([
                token for token, count in word_frequencies.items()
                if count < minimum_word_frequency
            ])
        if stopwords:
            exclude_tokens.update(stopwords)
       
        # Indexing
        for docid, tokens in tqdm(doc_data, total=doc_count):
            filtered_tokens = [
                token if token and token not in exclude_tokens else None
                for token in tokens
            ]
            index.add_doc(docid, filtered_tokens)
           
        index.get_statistics()
        return index


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''
    def add_doc(self, docid, tokens):
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1

    def save(self):
        print('Index saved!')
