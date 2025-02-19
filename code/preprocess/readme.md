# Data Preprocessing

This folder contains scripts for preprocessing three datasets: HotpotQA, MuSiQue, and TriviaQA. Each script is used to extract knowledge graphs (KGs) from the datasets. Below are the details of each script and how they are used:

## HotpotQA Dataset

`> python hotpot_extraction.py`

This script extracts triplets from the HotpotQA dataset. The main steps are as follows:
1. Load the dataset file `hotpot_dev_distractor_v1.json`.
2. Use the `llama3:8b` model to extract triplets from each context paragraph.
3. Save the extracted triplets to the specified output directory.



## MuSiQue Dataset

`> python musique_extraction.py`

This script extracts triplets from the MuSiQue dataset. The main steps are as follows:
1. Load the dataset file `musique_ans_v1.0_dev.jsonl`.
2. Use the `llama3:8b` model to extract triplets from each paragraph.
3. Save the extracted triplets to the specified output directory.

## TriviaQA Dataset

`> python trivia_extraction.py`

This script extracts triplets from the TriviaQA dataset. The main steps are as follows:
1. Load the dataset file `trivia.json`.
2. Use the `llama3:8b` model to extract triplets from each context paragraph.
3. Save the extracted triplets to the specified output directory.