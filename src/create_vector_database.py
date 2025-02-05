from datasets import load_dataset
from tqdm import tqdm

from langchain.docstore.document import Document as RawDocument
from langchain_core.documents.base import Document as SplitDocument
from datasets.arrow_dataset import Dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModel

import faiss


from config.config import (
    HF_DATASET_PATH,
    HF_EMBEDDING_MODEL,
    SEPARATOR_CHARS,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_split_texts(list_texts:list[str]) -> list[str]:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # The maximum number of characters in a chunk
        chunk_overlap=CHUNK_OVERLAP,  # The number of characters to overlap between chunks
        separators=SEPARATOR_CHARS,
    )

    split_text_idxs = []
    all_split_texts = []

    for text_idx, text in enumerate(list_texts):

        split_texts = text_splitter.split_text(text)

        all_split_texts += split_texts
        split_text_idxs += [text_idx]*len(split_texts)

    return all_split_texts, split_text_idxs



def cls_pooling(model_output:torch.Tensor) -> torch.Tensor:
    return model_output[:,0]

def get_embeddings(text_list:list[str], tokenizer:AutoTokenizer, model:AutoModel) -> torch.Tensor:

    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    
    model_output = model(**encoded_input)

    last_hidden_state = model_output.last_hidden_state

    cls_hidden_state = cls_pooling(last_hidden_state)

    return cls_hidden_state

def get_doc_texts(doc_dataset:Dataset, n_limit:int=0) -> list[str]:

    doc_texts = []

    for doc in doc_dataset:
        doc_texts.append(doc["text"])

        if len(doc_texts) == n_limit:
            break

    return doc_texts

def build_flat_index(index_embeddings:torch.Tensor):
    
    numpy_embeddings = index_embeddings.detach().cpu().numpy()

    # Get the dimension of the input embeddings.
    embedding_dim = numpy_embeddings.shape[1]

    # Build the index.
    index = faiss.IndexFlatL2(embedding_dim)

    # Add the embeddings to the index.
    index.add(numpy_embeddings)
    
    return index


def build_ivfflat_index(index_embeddings:torch.Tensor, n_centroids:int=5):

    embedding_dim = index_embeddings.shape[1]

    quantizer = faiss.IndexFlatL2(embedding_dim)  # the other index
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_centroids, faiss.METRIC_L2)

    numpy_embeddings = index_embeddings.detach().cpu().numpy()

    # Perform clustering to find the centroids.
    index.train(numpy_embeddings)

    # Add the embeddings to the index.
    index.add(numpy_embeddings)
    
    return index


def search_index(search_embedding:torch.Tensor, index, k):

    search_embedding = search_embedding.detach().cpu().numpy()

    distances, idxs = index.search(search_embedding, k=k)

    return distances[0], idxs[0]


def main():

    # Load the dataset from huggingface.
    doc_dataset = load_dataset(HF_DATASET_PATH, "raw_review_All_Beauty", trust_remote_code=True)['full']

    # Set a limit on the number of documents included in the index.
    n_docs = 60

    # Get the document texts.
    doc_texts = get_doc_texts(doc_dataset=doc_dataset, n_limit=n_docs)

    # Split the document texts into smaller texts. Return a list
    # with a document index for each split text piece that refers 
    # to which document the split text belongs too.
    split_texts, doc_idxs = get_split_texts(doc_texts)
    
    # Get the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(HF_EMBEDDING_MODEL)

    # Get the model.
    model = AutoModel.from_pretrained(HF_EMBEDDING_MODEL)

    # Get the embedding vectors that will be used to 
    # build the index.
    index_embeddings = get_embeddings(
        text_list=split_texts,
        tokenizer=tokenizer,
        model=model
    )

    # Build the index.
    # index = build_flat_index(
    #     index_embeddings=index_embeddings
    # )

    index = build_ivfflat_index(
        index_embeddings=index_embeddings,
        n_centroids=5
    )

    # Define a input search query for the index.
    search_query = "skin care routine reviews"

    # Create the input search embedding.
    search_embedding = get_embeddings(
        text_list=[search_query],
        tokenizer=tokenizer,
        model=model
    )

    # Search the index for k nearest neighbours.
    distances, idxs = search_index(
        search_embedding=search_embedding,
        index=index,
        k=3
    )

    # Print the results.
    for dist, idx in zip(distances, idxs):
        print(f"Distance: {dist}")
        print(f"Index text: {split_texts[idx]}")
        print(f"Doc text: {doc_texts[doc_idxs[idx]]}")




if __name__ == "__main__":
    main()