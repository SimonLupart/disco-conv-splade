# Download and extract the index files for the TopiocQA dataset
mkdir -p DATA/topiocqa_topics/
wget -O DATA/topiocqa_topics/raw_train.json https://zenodo.org/records/6151011/files/data/retriever/all_history/train.json?download=1
wget -O DATA/topiocqa_topics/raw_dev.json https://zenodo.org/records/6151011/files/data/retriever/all_history/dev.json?download=1

# Download and extract the index files for the TopiocQA dataset
wget -O DATA/full_wiki_segments_topiocqa.tsv https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv?download=1

