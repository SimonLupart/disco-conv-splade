# 🤖 DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search

This repository contains the code and resources for our **SIGIR 2025 full paper** by Lupart et al.


## 1. 🗂️ Installation and Dataset Download

### Conda Environment

```bash
conda env create -f environment.yml
conda activate cs
```

### Downloading TopiOCQA topics

We use the **TopioCQA** dataset for conversational passage retrieval.

```bash
wget -O raw_train.json https://zenodo.org/records/6151011/files/data/retriever/all_history/train.json?download=1
wget -O raw_dev.json https://zenodo.org/records/6151011/files/data/retriever/all_history/dev.json?download=1
````

### Download the Collection (Wikipedia Passages)

```bash
wget -O full_wiki_segments.tsv https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv?download=1
```

### Preprocessing

We will provide scripts to preprocess the TOPIOCQA conversation data into a format suitable for indexing and retrieval (queries, contexts, relevance labels, etc.).

*Coming soon: `scripts/preprocess_topiocqa.py`*



## 2. 🚀 Inference

We support two modes of inference: indexing the collection yourself or using a prebuilt index.

### Download a Prebuilt SPLADE Index

```bash
mkdir topiocqa_index
cd topiocqa_index
wget -O index.tar.gz https://surfdrive.surf.nl/files/index.php/s/TV9RLEYQqXA2Z04/download
tar -xzvf index.tar.gz
rm index.tar.gz
cd ..
```

### (Optional) Indexing with SPLADE yourself

You can build a SPLADE index over the TOPIOCQA passage collection.

```bash
# Example indexing command (adjust as needed for your SPLADE setup)
python -m splade.index \
  --input full_wiki_segments.tsv \
  --output_dir topiocqa_index/ \
  --model_name naver/splade-cocondenser-ensembledistil
```

### Retrieval with DiSCo

We will release retrieval scripts that allow running inference with DiSCo using our trained models.

*Coming soon: `disco/retrieve.py`*

You will also be able to run inference using different models available on HuggingFace, on all datasets evaluated in the paper.


## 3. 🚀 Training

We distill knowledge from large LLMs (e.g. LLaMA, Mistral) into a sparse retriever via multi-teacher distillation.

### Distillation on TopiOCQA

We will provide training scripts using combinations of LLM teachers:

* LLaMA
* Mistral
* More to come

*Coming soon: `disco/train_distill.py`*


## 4. All Pretrained Models

You can find all trained models on HuggingFace:

* DiSCo distilled from Mistral on TOPIOCQA
* DiSCo distilled from LLaMA & Mistral on TOPIOCQA
* DiSCo distilled from Human on QReCC
* DiSCo distilled from Human and Mistral on QReCC

See the Huggingface Collection [disco-splade-conv](https://huggingface.co/collections/slupart/splade-conversational-6800f23d0c61997aa33cf4e4) for more resources.

This code can also be adapted to train DiSCo on QReCC and do inference on the TREC CAsT and iKAT datasets.

## 5. 🙏 Acknowledgments

This work builds on and would not be possible without the following open-source contributions:

* [SPLADE](https://github.com/naver/splade) by Naver Labs Europe
* [TOPIOCQA](https://github.com/prdwb/topiocqa)
* [QReCC](https://github.com/apple/ml-qrecc)
* HuggingFace 🤗 ecosystem

Please cite our SIGIR 2025 paper if you use this work.


## 6. 📜 License

This repository is released under the **MIT License**.
See the [LICENSE](./LICENSE) file for details.

