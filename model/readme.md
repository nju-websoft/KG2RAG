# Model Directory

This directory is intended to store the open-source models used in the project. Follow the instructions below to download the models.

## Download `llama3:8b`
1. Install [ollama](https://ollama.com/)
2. Download [llama3:8b](https://ollama.com/library/llama3) with ollama

## Download `bge-reranker-large`

Command to download with [hf-mirror](https://hf-mirror.com/):
```sh
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh BAAI/bge-reranker-large