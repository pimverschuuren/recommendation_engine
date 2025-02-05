from datasets import load_dataset

from config.config import (
    LOCAL_DATASET_PATH,
    HF_DATASET_PATH
)



hf_dataset = load_dataset(HF_DATASET_PATH, "raw_review_All_Beauty", trust_remote_code=True)

print(type(hf_dataset))
print(type(hf_dataset['full']))
print(hf_dataset['full'][0])
#hf_dataset.save_to_disk(LOCAL_DATASET_PATH)
