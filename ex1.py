#%%
import pandas as pd
df = pd.read_excel('posts_first_targil.xlsx', sheet_name=None)
#%%
for sheet_name, sheet_df in df.items():
    print(f"Sheet name: {sheet_name}")
    print(sheet_df.columns)
#%%
# we're renaming J-P column to match others
df['J-P'].rename(columns={'Body': 'Body Text'}, inplace=True)
#%%
for sheet_name, sheet_df in df.items():
    print(f"Sheet name: {sheet_name}")
    print(sheet_df.columns)
#%%
import re
#%%
def clean_text(text):
    border_pattern = r"((?<!\w)[^\s\w]|[^\s\w](?!\w))"
    # Regex pattern
    dot_pattern = r"(?<!\w)([a-zA-Z]{2,})\.([a-zA-Z]{2,})(?!\w)"

    # Apply regex substitution
    cleaned_text = re.sub(dot_pattern, r"\1 . \2", text)

    cleaned_text = re.sub(border_pattern, r" \1 ", cleaned_text)

    # Replace multiple spaces with a single space
    return re.sub(r"\s+", " ", cleaned_text).strip()
#%%
for sheet_name, sheet_df in df.items():
    sheet_df = sheet_df.map(lambda x: clean_text(x) if isinstance(x, str) else x)
    sheet_df.to_csv(f'clean_data\\{sheet_name}.csv', index=False)
#%% md
# 2. Lemmatization
#%%
from nltk import word_tokenize, WordNetLemmatizer
#%%
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = word_tokenize(text)  # Tokenize text
    return " ".join([lemmatizer.lemmatize(word) for word in words])
#%%
import os

for file in os.listdir('./clean_data'):
    print(file)
    sheet = pd.read_csv(f'clean_data\\{file}')
    sheet = sheet.map(lambda x: lemmatize_text(x) if isinstance(x, str) else x)
    sheet.to_csv(f"lemmatize_data\\{file}", index=False)


#%% md
# BM25
#%%
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
import nltk
from scipy.sparse import save_npz
# nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

min_threshold = 5
#%%
def save_metadata(word_index, output_file):
    # save for feature use
    # Convert the dictionary to a DataFrame
    metadata_df = pd.DataFrame(list(word_index.items()), columns=["Word", "Index"])

    # Save the DataFrame to a CSV file
    metadata_df.to_csv(output_file, index=False)
    print(f"Metadata saved to {output_file}")

#%%
from collections import defaultdict

def get_word_freq(corpus):
    word_frequency = defaultdict(int)

    # Count occurrences
    for document in corpus:
        unique_words = set(document)  # Use a set to avoid counting duplicates in the same document
        for word in unique_words:
            word_frequency[word] += 1

    return  word_frequency
#%%
def remove_stopwords(text):
    words = text.split()  # Tokenize the text
    return [word.lower() for word in words if word.lower() not in stop_words]
#%% md
# bm25 to the lemmatized docs
#%%
for file in os.listdir('lemmatize_data'):
    print(file)
    sheet = pd.read_csv(f'lemmatize_data\\{file}')

    # Build the corpus
    if file == 'A-J.csv':
        corpus = [
            remove_stopwords(f'{row["title"]} {row["sub_title"]} {row["Body Text"]}')
            for _, row in sheet.iterrows()
        ]
    else:
        corpus = [
            remove_stopwords(f'{row["title"]} {row["Body Text"]}')
            for _, row in sheet.iterrows()
        ]

    bm25 = BM25Okapi(corpus)
    word_frequency = get_word_freq(corpus)
    all_words = [word for word in bm25.idf.keys() if word_frequency[word] > min_threshold]
    word_index = {word: idx for idx, word in enumerate(all_words)}
    save_metadata(word_index, f'metadata\\lemma\\{file}')

    rows, cols, data = [], [], []
    for word in all_words:
        scores = bm25.get_scores(word)
        for doc_idx, score in enumerate(scores):
            if score > 0:  # Only include non-zero scores to keep it sparse
                rows.append(doc_idx)  # Document index
                cols.append(word_index[word])  # Word index
                data.append(score)  # BM25 score

    sparse_bm25_matrix = csr_matrix((data, (rows, cols)), shape=(len(corpus), len(all_words)))

    save_npz(f'bm25\\lemma\\{file.split(".")[0]}', sparse_bm25_matrix)
#%% md
# bm25 to the clean docs
#%%
for file in os.listdir('clean_data'):
    print(file)
    sheet = pd.read_csv(f'clean_data\\{file}')

    # Build the corpus
    if file == 'A-J.csv':
        corpus = [
            remove_stopwords(f'{row["title"]} {row["sub_title"]} {row["Body Text"]}') for _, row in sheet.iterrows()
        ]
    else:
        corpus = [
            remove_stopwords(f'{row["title"]} {row["Body Text"]}') for _, row in sheet.iterrows()
        ]

    bm25 = BM25Okapi(corpus)
    word_frequency = get_word_freq(corpus)
    all_words = [word for word in bm25.idf.keys() if word_frequency[word] >= min_threshold]
    word_index = {word: idx for idx, word in enumerate(all_words)}
    save_metadata(word_index, f'metadata\\clean\\{file}')

    rows, cols, data = [], [], []
    for word in all_words:
        scores = bm25.get_scores(word)
        for doc_idx, score in enumerate(scores):
            if score > 0:  # Only include non-zero scores to keep it sparse
                rows.append(doc_idx)  # Document index
                cols.append(word_index[word])  # Word index
                data.append(score)  # BM25 score


    sparse_bm25_matrix = csr_matrix((data, (rows, cols)), shape=(len(corpus), len(all_words)))

    save_npz(f'bm25\\clean\\{file.split(".")[0]}', sparse_bm25_matrix)
#%% md
# 3: IG
#%%
from scipy.sparse import load_npz
import numpy as np
from sklearn.feature_selection import mutual_info_classif

#%% md
# IG for lemmatized
#%%
for file in os.listdir('bm25\\lemma'):
    # Load the sparse BM25 matrix
    sparse_bm25_matrix = load_npz(f'bm25\\lemma\\{file}')

    # Sum the BM25 scores for each word across all documents (proxy for importance)
    word_scores = np.array(sparse_bm25_matrix.sum(axis=0)).flatten()  # Sum along columns

    # Load word-to-index mapping
    word_metadata = pd.read_csv(f"metadata\\lemma\\{file.split('.')[0]}.csv")

    # Map scores to words
    df_word_scores = pd.DataFrame({
        "Word": word_metadata["Word"],
        "Score": word_scores
    }).sort_values(by="Score", ascending=False)

    # Save to CSV
    df_word_scores.to_csv(f"IG\\lemma\\{file.split('.')[0]}.csv", index=False)

    # Display top words by score
    print(df_word_scores.head())
#%% md
# IG for clean
#%%
for file in os.listdir('bm25\\clean'):
    # Load the sparse BM25 matrix
    sparse_bm25_matrix = load_npz(f'bm25\\clean\\{file}')

    # Sum the BM25 scores for each word across all documents (proxy for importance)
    word_scores = np.array(sparse_bm25_matrix.sum(axis=0)).flatten()  # Sum along columns

    # Load word-to-index mapping
    word_metadata = pd.read_csv(f"metadata\\clean\\{file.split('.')[0]}.csv")

    # Map scores to words
    df_word_scores = pd.DataFrame({
        "Word": word_metadata["Word"],
        "Score": word_scores
    }).sort_values(by="Score", ascending=False)

    # Save to CSV
    df_word_scores.to_csv(f"IG\\clean\\{file.split('.')[0]}.csv", index=False)

    # Display top words by score
    print(df_word_scores.head())
#%% md
# Word2Vec
#%%
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Load Google's pre-trained Word2Vec model (requires downloading)
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
#%%
def preprocess_text(text):
    # Remove punctuation and numbers
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\d+", "", text)      # Remove digits
    # Tokenize the text
    return text
#%%
for file in os.listdir('lemmatize_data'):
    print(file)
    sheet = pd.read_csv(f'lemmatize_data\\{file}')

    if file == 'A-J.csv':
        corpus = [
            remove_stopwords(preprocess_text(f'{row["title"]} {row["sub_title"]} {row["Body Text"]}')) for _, row in sheet.iterrows()
        ]
    else:
        corpus = [
            remove_stopwords(preprocess_text(f'{row["title"]} {row["Body Text"]}')) for _, row in sheet.iterrows()
        ]

    docs_matrix = []
    for doc_idx, row in enumerate(corpus):
        matrix = []
        for word in row:
            if word in model:
                matrix.append(model[word].tolist())
            else:
                matrix.append([0] * model.vector_size)  # Placeholder for missing words

        docs_matrix.append(np.array(matrix).mean(axis=0))
    pd.DataFrame(docs_matrix).to_csv(f'Word2Vec\\lemma\\{file}', index=False)



#%%
for file in os.listdir('clean_data'):
    print(file)
    sheet = pd.read_csv(f'clean_data\\{file}')

    if file == 'A-J.csv':
        corpus = [
            remove_stopwords(preprocess_text(f'{row["title"]} {row["sub_title"]} {row["Body Text"]}')) for _, row in sheet.iterrows()
        ]
    else:
        corpus = [
            remove_stopwords(preprocess_text(f'{row["title"]} {row["Body Text"]}')) for _, row in sheet.iterrows()
        ]
    docs_matrix = []
    for doc_idx, row in enumerate(corpus):
        matrix = []
        for word in row:
            if word in model:
                matrix.append(model[word].tolist())
            else:
                matrix.append([0] * model.vector_size)  # Placeholder for missing words

        docs_matrix.append(np.array(matrix).mean(axis=0))

    pd.DataFrame(docs_matrix).to_csv(f'Word2Vec\\clean\\{file}', index=False)


#%% md
# Doc2Vec on original
#%%
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

#%%
original = df.copy()
corpus = []
j = 0
for sheet_name, sheet_df in original.items():
    # Prepare the corpus as a list of TaggedDocument objects
    if sheet_name == 'A-J':
        corpus += [
            TaggedDocument(words=f'{row["title"]} {row["sub_title"]} {row["Body Text"]}'.split(), tags=[str(i+j*1000)])
            for i, row in sheet_df.iterrows()
        ]
    else:
        corpus += [
            TaggedDocument(words=f'{row["title"]} {row["Body Text"]}'.split(), tags=[str(i+j*1000)])
            for i, row in sheet_df.iterrows()
        ]
    j+=1

model = Doc2Vec(documents=corpus, vector_size=300, window=5, min_count=5, workers=4)

document_vectors = [model.dv[tag] for tag in model.dv.index_to_key]

# Save document vectors to a CSV file
document_vectors = pd.DataFrame(document_vectors)
document_vectors.iloc[0:599].to_csv("Doc2Vec\\A-J.csv", index=False)
document_vectors.iloc[599:1148].to_csv("Doc2Vec\\BBC.csv", index=False)
document_vectors.iloc[1148:1747].to_csv("Doc2Vec\\J-P.csv", index=False)
document_vectors.iloc[1747:2346].to_csv("Doc2Vec\\NY-T.csv", index=False)
#%% md
# BERT
#%%
from transformers import BertTokenizer, BertModel
import torch
#%%
# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

for sheet_name, sheet_df in original.items():
    # Prepare the corpus as a list of TaggedDocument objects
    if sheet_name == 'A-J':
        corpus = [
            f'{row["title"]} {row["sub_title"]} {row["Body Text"]}'.lower() for _, row in sheet_df.iterrows()
        ]
    else:
        corpus = [
            f'{row["title"]} {row["Body Text"]}'.lower() for _, row in sheet_df.iterrows()
        ]

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    # model = model.to(device)
    # Tokenize and encode the corpus
    inputs = tokenizer(corpus, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract token embeddings from the last hidden state
    last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

    # Aggregate token embeddings to get sentence/document-level embeddings (e.g., mean pooling)
    corpus_embeddings = torch.mean(last_hidden_state, dim=1)  # Shape: [batch_size, hidden_size]

    # Convert to numpy array for easier manipulation
    corpus_embeddings = corpus_embeddings.numpy()

    print(corpus_embeddings.shape)  # (num_documents, hidden_size)

    pd.DataFrame(corpus_embeddings).to_csv(f"Bert\\{sheet_name}.csv", index=False)
#%%
from sentence_transformers import SentenceTransformer

# Load pre-trained Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

for sheet_name, sheet_df in original.items():
    # Prepare the corpus as a list of TaggedDocument objects
    if sheet_name == 'A-J':
        corpus = [
            f'{row["title"]} {row["sub_title"]} {row["Body Text"]}' for _, row in sheet_df.iterrows()
        ]
    else:
        corpus = [
            f'{row["title"]} {row["Body Text"]}' for _, row in sheet_df.iterrows()
        ]

    # Generate embeddings for the corpus
    corpus_embeddings = model.encode(corpus)

    # Check the shape of the embeddings
    print(corpus_embeddings.shape)  # (num_documents, embedding_size)

    pd.DataFrame(corpus_embeddings).to_csv(f"Bert-Sentence\\{sheet_name}.csv", index=False)