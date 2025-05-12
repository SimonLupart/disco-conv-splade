# 🤖 [SIGIR 2025] DiSCo: LLM Knowledge Distillation for Efficient Sparse Retrieval in Conversational Search

This repository contains the code and resources for our **SIGIR 2025 full paper** DiSCo for Conversational Search by Lupart et al. It is based on the SPLADE github by Naver [[link]](https://github.com/naver/splade), using a huggingface trainer for training, and the default code for index and retrieval.

We provide below an example of usage of the github for training, indexing and retrieval on TopiOCQA.

## 1. 🗂️ Installation and Dataset Download

### Conda Environment

```bash
conda env create -f environment.yml
conda activate disco
```

### Downloading TopiOCQA topics and the Wikipedia Passages Collection

We use the **TopioCQA** dataset for conversational passage retrieval.

```bash
bash setup_script/dl_topiocqa.sh
```

### Preprocessing

We provide scripts to preprocess the TopiOCQA conversation data into a format suitable for indexing and retrieval (queries, contexts, relevance labels, etc.). By default we use row numbers as id instead of the original conv_turn format.

```bash
python setup_script/parse_topiocqa.py
```


## 2. 🚀 Inference

We support two modes of inference: 1. (Recommended) Using our prebuilt index, indexed with the checkpoint 
`naver/splade-cocondenser-ensembledistil`; OR 2. Indexing the collection yourself.

### Download a Prebuilt SPLADE Index

```bash
bash setup_script/dl_index_topiocqa.sh
```

### (Optional) You can indexing the TopiOCQA collection with SPLADE yourself

You can build a SPLADE index over the TopiOCQA passage collection:

```bash
config=disco_topiocqa_mistral_llama.yaml
collection_path=DATA/full_wiki_segments_topiocqa.tsv
index_dir=DATA/topiocqa_index_self

python -m splade.index --config-name=$config \
    init_dict.model_type_or_dir=naver/splade-cocondenser-ensembledistil \
    config.pretrained_no_yamlconfig=true \
    config.hf_training=false \
    config.index_dir="$index_dir" \
    data.COLLECTION_PATH="$collection_path" \
    config.index_retrieve_batch_size=128
```

### Retrieval with DiSCo

Retrieval script using one of our DiSCo HuggingFace checkpoint:

```bash
mkdir -p EXP/checkpoint_exp/

config=disco_topiocqa_mistral_llama.yaml
index_dir=DATA/topiocqa_index
out_dir=EXP/checkpoint_exp/disco_TOPIOCQA_mistral_llama_out_hf/

python -m splade.retrieve --config-name=$config \
    init_dict.model_type_or_dir_q=slupart/splade-disco-topiocqa-mistral \
    config.pretrained_no_yamlconfig=true \
    config.hf_training=false \
    config.index_dir="$index_dir" \
    config.out_dir=$out_dir
```

You can also run inference using different models available on HuggingFace, with the models we trained on TopiOCQA:

* `slupart/splade-disco-topiocqa-mistral`
* `slupart/splade-disco-topiocqa-llama-mistral`


## 3. 🚀 Training

In DiSCO, we distill knowledge from LLMs (e.g. LLaMA, Mistral) into a sparse retriever via multi-teacher distillation.

### Distillation on TopiOCQA

First download the distillation file for TopiOCQA, from Mistral and Llama

```bash
bash setup_script/dl_distillation_topiocqa.sh
```

Then train the DiSCo model using the distillation file as teacher.

```bash
port=$(shuf -i 29500-29599 -n 1)

config=disco_topiocqa_mistral_llama.yaml
runpath=DATA/topiocqa_distil/distil_run_top_llama_mistral.json
ckpt_dir=EXP/checkpoint_exp/disco_TOPIOCQA_mistral_llama/

torchrun --nproc_per_node 1 --master_port $port -m splade.hf_train \
    --config-name=$config  \
    data.TRAIN.DATASET_PATH=$runpath \
    config.checkpoint_dir=$ckpt_dir
```

Similarly you can evaluate this model:

```bash
config=disco_topiocqa_mistral_llama.yaml
ckpt_dir=EXP/checkpoint_exp/disco_TOPIOCQA_mistral_llama/
index_dir=DATA/topiocqa_index
out_dir=EXP/checkpoint_exp/disco_TOPIOCQA_mistral_llama_out/

python -m splade.retrieve \
    --config-name=$config \
    config.checkpoint_dir=$ckpt_dir \
    config.index_dir=$index_dir \
    config.out_dir=$out_dir
```

## 4. 🚀 Additional Resources

You can find all trained models on HuggingFace in our [disco-splade-conv](https://huggingface.co/collections/slupart/splade-conversational-6800f23d0c61997aa33cf4e4) Collection:

* Models trained on TopiOCQA and QReCC with different teachers
* Mistral Rewritten Queries on training sets of TopiOCQA and QReCC, used for the distillation
* Mistral Rewritten Queries on all test sets used as baselines (TopiOCQA, QReCC, TREC CAsT 2020, TREC CAsT 2022, TREC iKAT 2023)

This code can also be adapted to train DiSCo on QReCC and do inference on the TREC CAsT and iKAT datasets.

Snippet code for Training, Indexing and Retrieval can be found in `train.sh`, `index.sh` and `retrieve.sh`.

## 5. 🙏 Acknowledgments

This work builds on and would not be possible without the following open-source contributions:

* [SPLADE](https://github.com/naver/splade) by Naver Labs Europe
* [TopiOCQA](https://mcgill-nlp.github.io/topiocqa/)
* [QReCC](https://github.com/apple/ml-qrecc)
* HuggingFace 🤗 ecosystem

Feel free to contact us by email s.c.lupart@uva.nl

## 6. 📜 Citations

Please cite our SIGIR 2025 paper and the original SPLADE works if you use this work:
* SIGIR 2025 full paper, DiSCo
```
@article{lupart2024disco,
  title={DiSCo Meets LLMs: A Unified Approach for Sparse Retrieval and Contextual Distillation in Conversational Search},
  author={Lupart, Simon and Aliannejadi, Mohammad and Kanoulas, Evangelos},
  journal={arXiv preprint arXiv:2410.14609},
  year={2024}
}
```
* SIGIR22 short paper, SPLADE++ (v2bis)
```
@inproceedings{10.1145/3477495.3531857,
author = {Formal, Thibault and Lassance, Carlos and Piwowarski, Benjamin and Clinchant, St\'{e}phane},
title = {From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531857},
doi = {10.1145/3477495.3531857},
abstract = {Neural retrievers based on dense representations combined with Approximate Nearest Neighbors search have recently received a lot of attention, owing their success to distillation and/or better sampling of examples for training -- while still relying on the same backbone architecture. In the meantime, sparse representation learning fueled by traditional inverted indexing techniques has seen a growing interest, inheriting from desirable IR priors such as explicit lexical matching. While some architectural variants have been proposed, a lesser effort has been put in the training of such models. In this work, we build on SPLADE -- a sparse expansion-based retriever -- and show to which extent it is able to benefit from the same training improvements as dense models, by studying the effect of distillation, hard-negative mining as well as the Pre-trained Language Model initialization. We furthermore study the link between effectiveness and efficiency, on in-domain and zero-shot settings, leading to state-of-the-art results in both scenarios for sufficiently expressive models.},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2353–2359},
numpages = {7},
keywords = {neural networks, indexing, sparse representations, regularization},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```

## 7. License

This repository is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
See the [LICENSE](./LICENSE) file for details.

