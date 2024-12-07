# Information Retrieval Assignment 1 - README

## Students Information
- **Names:** Moshe Shahar, Yonatan Klein 
- **IDs:** 211692165, 322961764

## Overview
This project focuses on text processing and feature extraction techniques for information retrieval, leveraging the dataset in `posts_first_targil.xlsx`. The assignment includes:
1. Data preprocessing (text cleaning and lemmatization).
2. Application of BM25 for ranking.
3. Feature selection using Information Gain (IG).
4. Vectorization techniques: Word2Vec, Doc2Vec, BERT, and Sentence-BERT.

## Methodology

### 1. **Data Preprocessing**
- **Steps:**  
  - Renamed columns for consistency.
  - Applied regex patterns to clean text by removing special characters and normalizing text structure.
  - Stored cleaned data as CSV files.
  
- **Reasoning:**  
  Cleaning ensures uniformity across datasets and removes noise, enabling efficient feature extraction and model training.

### 2. **Lemmatization**
- **Steps:**  
  - Tokenized text and applied lemmatization using NLTK's WordNet lemmatizer.
  - Stored lemmatized outputs in a separate directory.

- **Reasoning:**  
  Lemmatization reduces words to their base form, reducing dimensionality without losing semantic meaning.

### 3. **BM25 Ranking**
- **Steps:**  
  - Constructed document-term matrices using BM25 on both cleaned and lemmatized datasets.
  - Incorporated stopword removal and threshold-based filtering for feature selection.
  - Sparse matrices were stored in `.npz` format for further use.

- **Reasoning:**  
  BM25 is effective for ranking documents based on relevance and capturing term frequency nuances.

### 4. **Feature Selection (Information Gain)**
- **Steps:**  
  - Analyzed BM25 matrices for feature importance.
  - Identified top words by their cumulative BM25 scores and stored metadata.

- **Reasoning:**  
  Feature selection refines the representation by emphasizing informative terms, optimizing model performance.

### 5. **Vector Representations**
- **Word2Vec:** Generated word embeddings using Google's pre-trained model. Averaged word vectors per document.  
- **Doc2Vec:** Tagged documents and trained a model to create document embeddings.  
- **BERT & Sentence-BERT:** Utilized pre-trained transformers to generate contextual embeddings for documents.  

- **Reasoning:**  
  These techniques enable semantic representation of text for downstream tasks like clustering and classification.

## Results and Analysis

### Observations
1. **BM25 Ranking:**  
   - Lemmatized data yielded slightly better term relevance due to reduced sparsity.
   - Thresholding improved computational efficiency by removing low-frequency words.

2. **Vectorization:**  
   - **Word2Vec:** Effective for capturing word semantics but struggled with out-of-vocabulary terms.  
   - **Doc2Vec:** Performed well on small datasets due to its ability to capture contextual meaning.  
   - **BERT:** Offered the most comprehensive representations, albeit computationally expensive.

### Errors and Challenges
- **Cleaning Errors:** Some regex patterns occasionally split valid tokens (e.g., abbreviations).  
- **Sparse Matrices:** High sparsity in BM25 matrices for large vocabulary sizes.  
- **BERT Limitations:** Token length truncation affected longer documents.  

### Insights
- Preprocessing significantly impacts model performance; lemmatized datasets often perform better.  
- Word2Vec is efficient for lightweight applications, while BERT embeddings are ideal for high-accuracy tasks.  
- Threshold-based feature selection is crucial for balancing relevance and computational cost.

## How to Run the Code
1. Place `posts_first_targil.xlsx` in the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the script:
   ```bash
   python ex1.py
