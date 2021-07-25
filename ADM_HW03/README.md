# ADM-HW3

**Goal of the homework**: Build a search engine over the "best books ever" list of GoodReads. Unless differently specified, all the functions must be implemented from scratch.

## 1. Data Collection
In order to complete the tasks, we needed to build our own dataset collecting the urls of the books listed in the first 300 pages of the [best books ever list](https://www.goodreads.com/list/show/1.Best_Books_Ever?page=1). Then we were asked to download the html corresponding to each of the collected urls and organize them in folders by page. Finally, we parsed the downloaded pages extracting interesting books informations.

## 2. Search Engine
Now, we want to create two different Search Engines that, given as input a query, return the books that match the query.

### 2.1.  Conjunctive Query
For the first version of the search engine, we narrow our interest on the Plot of each document. It means that you will evaluate queries only with respect to the book's plot.

### 2.2. Conjunctive Query & Ranking Score
For the second search engine, given a query, we want to get the top-k (the choice of k it's up to you!) documents related to the query. In particular:

- Find all the documents that contains all the words in the query.
- Sort them by their similarity with the query
- Return in output k documents, or all the documents with non-zero similarity with the query when the results are less than k. You must use a heap data structure (you can use Python libraries) for maintaining the top-k documents.

To solve this task, you will have to use the tfIdf score, and the Cosine similarity. The field to consider it is still the plot.

## 3. Define a new score!
Build a new metric to rank books based on the queries of their users.

In this scenario, a single user can give in input more information than the single textual query, so you need to take into account all this information, and think a creative and logical way on how to answer at user's requests.

## 4. Make a nice visualization!
Our goal is to quantify and visualize the writers' production.

1. Consider the first 10 BookSeries in order of appearance.
2. Build a 2-d plot where the x-axis is the years since publication of the first book (starting from 0), and y-axis there must be the cumulative series page count (all the Series start from (0,num_pages) point, which represents the first book). Since we want the cumulative number of page, the y-axis value of each book is added to the previous point.

## 5. Algorithmic Question

You are given a string written in english capital letters, for example S="CADFECEILGJHABNOPSTIRYOEABILCNR." You are asked to find the maximum length of a subsequence of characters that is in alfabetical order. For example, here a subsequence of characters in alphabetical order is the "ACEGJSTY": "CADFECEILGJHABNOFPSTIRYOEABILCNR." Among all the possible such sequences, you are asked to find the one that is the longest.

# Structure of the repository

## Usage:
- `python get_booklist.py`: download urls and saves them into the file `books`
- `python crawler.py [--n_threads N_THREADS] [--start START] [--end END]`: starts `N_THREADS` threads to download from `START` to `END` books taking url from the file `books`
- `python build_dataset.py`: reads the html files preprocessing their content and saves it into a `dataset.tsv` file

## Directory descriptions:

1. **`books`**:
> a list of url of the first 300 most read books on [GoodReads](https://www.goodreads.com/)
2. **`vocabulary`**, **`index`** and **`tfidf_index`**:
> pickle dictionaries of each data structure
3. **`dataset.tsv`**:
> records of each book with each respective tag content appropriately parsed
4. **`functions.py`**:
> all functions used in the notebook and in the scripts
5. **`get_booklist.py`**:
> download the urls of the first 300 pages of most read books from GoodReads
6. **`crawler.py`**:
> downloads the urls in `books` and saves the content in html files as `article_[i].html` organizing them into directors `page_[i]/`
7. **`build_dataset.py`**:
> creates `dataset.tsv` from all documents contained in `page_[i]/` subdirectories named `article_[i].html`
8. **`longest_subsequence.py`**:
> contains both the recursive and dynamic programming solution to the longest increasing subsequence problem, they're also shown in the notebook along with the running time analysis
