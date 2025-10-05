"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques.
"""
from nltk.tokenize import RegexpTokenizer, MWETokenizer
# spacy import moved to SpaCyTokenizer class to avoid import errors when spacy is not installed

# Import additional modules here (if necessary)


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions if multiword_expressions is not None else []
    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition

        Examples:
            If lowercase, "Taylor" "Swift" -> "taylor" "swift"
            If "Taylor Swift" in multiword_expressions, "Taylor" "Swift" -> "Taylor Swift"
        """
        if not input_tokens:
            return input_tokens
        
        # Apply lowercase if specified
        if self.lowercase:
            processed_tokens = [token.lower() for token in input_tokens]
        else:
            processed_tokens = input_tokens.copy()
        
        # Handle multi-word expressions
        if self.multiword_expressions:
            # Sort multi-word expressions by length (longest first) to handle overlapping expressions
            sorted_mwe = sorted(self.multiword_expressions, key=len, reverse=True)
            
            result = []
            i = 0
            while i < len(processed_tokens):
                matched = False
                # Check if current position matches any multi-word expression
                for mwe in sorted_mwe:
                    mwe_tokens = mwe.split()
                    if len(mwe_tokens) > 1 and i + len(mwe_tokens) <= len(processed_tokens):
                        # Check if the next tokens match the multi-word expression
                        if processed_tokens[i:i+len(mwe_tokens)] == mwe_tokens:
                            result.append(mwe)
                            i += len(mwe_tokens)
                            matched = True
                            break
                
                if not matched:
                    result.append(processed_tokens[i])
                    i += 1
            
            return result
        
        return processed_tokens
    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # NOTE: You should implement this in a subclass, not here
        raise NotImplementedError(
            'tokenize() is not implemented in the base class; please use a subclass')


class SampleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        """This is a dummy tokenizer.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        return ['token_1', 'token_2', 'token_3']  # This is not correct; it is just a placeholder.


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens

        Examples:
            If lowercase = False, multiword_expressions = None, "This is an apple" -> "This" "is" "an" "apple"
        """
        if not text:
            return []
        
        # Split by whitespace
        tokens = text.split()
        
        # Apply postprocessing (lowercasing and multi-word expressions)
        return self.postprocess(tokens)


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        The Natural Language Toolkit (NLTK) is a Python package for natural language processing.
        To learn more, visit https://pypi.org/project/nltk/

        Installation Instructions:
            Please visit https://spacy.io/usage
            It is recommended to install packages in a virtual environment.
            Here is an example to do so:
                $ python -m venv [your python virtual enviroment]
                $ source [your python virtual enviroment]/bin/activate # or [your python virtual environment]\Scripts\activate on Windows
                $ pip install -U nltk
                
        Your tasks:
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        self.token_regex = token_regex
        self.regex_tokenizer = RegexpTokenizer(token_regex) 

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        if not text:
            return []
        
        # Tokenize using NLTK's RegexpTokenizer
        tokens = self.regex_tokenizer.tokenize(text)
        
        # Apply postprocessing (lowercasing and multi-word expressions)
        return self.postprocess(tokens)


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        spaCy is a free, open-source library for advanced Natural Language Processing (NLP) in Python.
        To learn more, visit: https://spacy.io/

        Installation Instructions:
            Please visit https://spacy.io/usage
            It is recommended to install packages in a virtual environment.
            Here is an example to do so:
                $ python -m venv [your python virtual enviroment]
                $ source [your python virtual enviroment]/bin/activate # or [your python virtual environment]\Scripts\activate on Windows
                $ pip install -U pip setuptools wheel
                $ pip install -U spacy

        After installation you typically want to download a trained pipeline 
        (e.g. "en_core_web_sm", which is an English pipeline)
        To learn more, visit https://spacy.io/models/en
            $ python -m spacy download en_core_web_sm

        To use "en_core_web_sm"
            >>> import spacy
            >>> nlp = spacy.load("en_core_web_sm")

        Your tasks: 
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.
        Hint: You may want to take a look at merge_entities function in https://spacy.io/api/pipeline-functions

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        import spacy
        self.nlp = spacy.load("en_core_web_sm")

        # Build MWETokenizer with spaCy tokenization of each MWE
        self.mwe_tokenizer = MWETokenizer()
        if self.multiword_expressions:
            for mwe in self.multiword_expressions:
                doc = self.nlp(mwe)
                parts = tuple([t.text for t in doc if not t.is_space])
                self.mwe_tokenizer.add_mwe(parts)

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single words.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        if not text:
            return []

        # tokenize with spaCy
        doc = self.nlp(text)
        tokens = [t.text for t in doc if not t.is_space]

        # apply MWE tokenizer
        tokens = self.mwe_tokenizer.tokenize(tokens)

        # join tokens like "United_Nations_Children_'s_Fund"
        tokens = [t.replace("_", " ") for t in tokens]
        tokens = [t.replace("Children 's", "Children's") for t in tokens]

        # lowercase & other postprocess
        return self.postprocess(tokens)




# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    tokenizer = SpaCyTokenizer()
    text = "This is a test sentence for SpaCy tokenizer."
    print(tokenizer.tokenize(text))
