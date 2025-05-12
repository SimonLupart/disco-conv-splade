# Download and extract the SPLADE index
mkdir -p DATA/topiocqa_index
wget -O DATA/topiocqa_index/index.tar.gz https://surfdrive.surf.nl/files/index.php/s/TV9RLEYQqXA2Z04/download
tar -xzvf DATA/topiocqa_index/index.tar.gz -C DATA/topiocqa_index
rm DATA/topiocqa_index/index.tar.gz

