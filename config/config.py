
# Dataset
HF_DATASET_PATH = "McAuley-Lab/Amazon-Reviews-2023"
LOCAL_DATASET_PATH = "data/amazon_review_data.jsonl"



# Vector database
SEPARATOR_CHARS = [
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
    "\u200b",  # Zero-width space
    "\uff0c",  # Fullwidth comma
    "\u3001",  # Ideographic comma
    "\uff0e",  # Fullwidth full stop
    "\u3002",  # Ideographic full stop
    "",
]

CHUNK_SIZE = 100
CHUNK_OVERLAP = 10

HF_EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"