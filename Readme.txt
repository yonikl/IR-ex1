# README

## Student Details
- **Names**: Moshe Shahar & Yoni klein
- **IDs**: 211692165 322961764

---

## Overview of the Work
This document describes the work we carried out, the methods we used, and the rationale behind our decisions. It also presents the results, discusses any errors encountered, and highlights the insights gained throughout the process.

---

## Explanation of the Process

### 1. **Problem Understanding**
We aimed to [describe the main objective of your task briefly, e.g., "implement and analyze the BM25 algorithm for document ranking"]. To accomplish this, we explored different approaches to preprocessing, implementation, and evaluation of the algorithm.

### 2. **Our Approach**
- **Preprocessing**:
  - We used a corpus of documents stored in CSV files and cleaned the data by:
    - Removing stop words.
    - Lemmatizing terms to ensure consistency.
    - Tokenizing the text.
- **Implementation**:
  - We leveraged the `rank_bm25` library for efficient implementation of the BM25 ranking algorithm. The repository used as a reference for the library was: [Rank BM25 GitHub Repo](https://github.com/dorianbrown/rank_bm25/tree/master).
  - The documents were transformed into BM25 vectors using a sparse matrix representation for efficient computation.
- **Analysis**:
  - To evaluate the results, we computed metrics such as [add any specific metric, e.g., precision, recall, Information Gain].

---

## Rationale Behind Decisions
1. **Why BM25**:
   - BM25 is a widely recognized ranking algorithm that balances term frequency and inverse document frequency. Its ability to prioritize relevant terms made it a strong candidate.

2. **Sparse Matrix Representation**:
   - Given the size of our corpus and the high sparsity of BM25 scores, we chose a sparse matrix representation to save memory and computational resources.

3. **Using the `rank_bm25` Library**:
   - This library is well-maintained and offers a robust implementation of BM25, saving us development time while ensuring correctness.

4. **Preprocessing Choices**:
   - **Stop Word Removal**: To eliminate commonly occurring but uninformative words.
   - **Lemmatization**: To reduce words to their root forms and improve matching across similar terms.

5. **Evaluation and Feature Selection**:
   - We evaluated the relevance of features (words) using Information Gain to better understand the impact of different terms on the results.

---

## Results and Analysis

### Results:
1. **BM25 Vectorization**:
   - The sparse matrix representation reduced memory usage significantly, allowing us to process a large corpus efficiently.
2. **Top Informative Words**:
   - Using Information Gain, the most significant words were [insert examples of top words].
3. **Performance Metrics**:
   - The model achieved [insert results] for [insert evaluation metric].

### Errors and Challenges:
- **Data Quality Issues**:
  - Some documents contained missing or irrelevant data, which required additional cleaning.
- **Sparse Data**:
  - The high sparsity of the data made it challenging to ensure meaningful BM25 scores for every word.

---

## Insights and Conclusions

1. **Key Insights**:
   - The importance of preprocessing cannot be overstated, as it directly impacts the quality of vectorization and ranking.
   - Sparse matrix techniques are indispensable when dealing with high-dimensional data.

2. **Conclusions**:
   - BM25 proved to be an effective method for ranking documents, and the integration of Information Gain helped us identify the most relevant features.

---

## Workflow and Rational Decision-Making
1. **Initial Research**:
   - We started by reviewing the `rank_bm25` repository and researching best practices for BM25 implementation and evaluation.

2. **Preprocessing Decisions**:
   - Based on online resources and trial-and-error, we chose a minimal preprocessing pipeline to balance efficiency and accuracy.

3. **Sparse Matrix Representation**:
   - Online examples and the structure of the `rank_bm25` library guided us toward using sparse matrices to handle our dataset.

4. **Evaluation and Iteration**:
   - After implementing BM25, we iteratively refined the feature selection and preprocessing steps based on our evaluation results.

---

This document summarizes our approach and highlights the reasoning behind our decisions, as well as the results and insights gained from our work. We believe that the thoroughness of our process ensured a comprehensive understanding of the BM25 algorithm and its practical application.