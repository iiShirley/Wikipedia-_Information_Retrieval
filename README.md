With the files we provide in the `HW1.zip` file, you will launch a search engine (that doesn’t work as expected yet) from a server. Your goal in this homework is to make the search engine work as expected, i.e. returning a list of relevant results after entering different search queries. This document guides you to start the server.

## Code Structure
### Files

The following files are the tools to run the search engine. You don't need to modify them. 

- app.py: The main file that contains the FastAPI code. 
- models.py: The file that contains some data classes that are used in the project. 

The following files **REQUIRES** your implementation. 

- `document_preprocessor.py`: The file that contains the code to preprocess the documents. 
- `indexing.py`: The file that contains the code to index the documents. 
- `ranker.py`: The file that contains the code to rank the documents. 
- `relevance.py`: The file that contains the code to compute metrics like MAP and NDCG.
- `test/test_preprocessor_public.py`: The file that contains the test you should implement for the document preprocessor by yourself. (Problem 11 HW1) 

The following file is the pipeline that runs the search engine. You may need to make changes if you want to add more functionalities. 

- `pipeline.py`: This file runs the search engine pipeline. You can modify it or write your own main function to execute the engine for the written homework questions.

### Data

You shold download the following files from canvas, and store them in the data folder for easy access. 
- data  
    - `multi_word_expressions.txt`: The file that contains the multi-word expressions. 
    - `wikipedia_200k_dataset.jsonl.gz`: The file that contains the Wikipedia dataset. 
    - `stopwords.txt`: The file that contains the stopwords. 
    - `relevance.test.csv`: The file that contains the test data for the relevance scorer. 

The `wikipedia_200k_dataset.jsonl.gz` dataset contains 200 thousand Wikipedia articles. It is in JSONL format where each lines is a separate JSON of the following format. 

```
{
    "docid": <document id>
    "title": <document title>
    "text": <the entire text of the document>
    "categories": [<each Wikipedia category>]
}
```

### Unit Tests

The following folder contains the tests for the search engine.
- tests
    - `test_document_preprocessor_public.py`: The file that contains the tests for the document preprocessor. 
    - `test_indexing_public.py`: The file that contains the tests for the indexing.
    - `test_ranker_public.py`: Contains unit tests for individual scorers, verifying they produce the correct score for each document-query pair.
    - `test_relevance_scorers_public.py`: Contains integration tests with a larger dataset to validate the entire ranking and scoring system functions correctly.
    - `test_metrics_public.py`: Contains tests for evaluating the Mean Average Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG) metrics.

You can run these Python files directly to test your implementation. We require you to implement one of the test cases for the document preprocessor. You will find the detailed instructions in the file with a TODO comment. 

To run a test, first navigate to the tests directory, then execute `python test_xxx.py` in the terminal.

## How to run the Search Engine

This is a Python 3.11 FastAPI project with the necessary requirements added to the `requirements.txt`.

### Environment Setup

After downloading the `HW1.zip` file from canvas, you will unzip the file, which gives you a `HW1` folder. You will also download `multi_word_expressions.txt` from canvas. Put it in the `HW1` folder. 

Now, we will create a virtual environment for your search engine project.

1. Install Python 3.11

    Follow the link [Python 3.11](https://www.python.org/downloads/release/python-3112/) to install Python 3.11. After installation, you can check the version of your Python by running the following command.

    For macOS and Linux, you will be running this in Terminal.

    For Windows, you will be running this in Command Prompt.

    ```
    python --version
    ```

2. Create a virtual environment

    Get the path of your python installation.

    Use the following command to create a virtual environment specifically for SI 650.

    ```
    <path of your python installation>/python -m venv si650
    ```

    If you installed Python 3.11 without overwriting the default Python version, use the following command:

    ```
    python3.11 -m venv si650
    ```

    This will create a folder `si650` inside the folder you navigated to. In the example above, this will create a virtual environment (a folder) inside `my_project`.

3. Activate the environment 

    - For Windows, run
    ```
    si650/bin/activate
    ```

    - For Mac and others, run
    ```
    source si650/bin/activate
    ```

4. Install the requirements

    After activating the virtual environment, navigate to the `HW1` folder by using the cd command. 

    Run the following command:

    ```
    python -m pip install -r requirements.txt
    ```

    This will install the libraries you need to start the server.

    Note: `requirements.txt` is not an exhaustive list of packages available on Autograder. Many common Python packages, such as NumPy, Pandas, gzip, and math, are also available. If you need to confirm the availability of a specific package not listed, please contact the GSIs via Piazza.

### Start the Server Locally

After you have all of these files and the necessary Python requirements installed in your environment, run 

```
python3 -m uvicorn app:app
```

to start the server.

It will give you an address that starts with `http://`. Copy and paste it in your browser. 

NOTE: To get a functioning search engine, you need to implement at least one of the preprocessor, indexer, and ranker. Otherwise you would get an empty page with no search results. 


### Deactivate the virtual environment

After you are done running the server, you can use this command to deactivate the virtual environment:

```
deactivate
```

After running the deactivate command, your virtual environment will be deactivated, and you'll return to the global Python environment. You'll see that the virtual environment's name (in our case, `(si650)`) disappears from your command prompt, indicating that you are no longer in the virtual environment.


More comments are present in the code files themselves. And if you have trouble understanding parts of the code, please ping any of the GSIs for the course on Slack or create a post on Piazza. 
