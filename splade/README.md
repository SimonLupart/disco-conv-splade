# DiSCo Conversational Search

This section of the code relies on the [SPLADE GitHub repository](https://github.com/naver/splade) from the original SPLADE authors, Formal et al. We only truncated all unnecessary functions in the context of this work. This `README.md` file is specific to the SPLADE repository and outlines the organization of the various files contained within.

### Repository Overview

- **Training**:
  The training of SPLADE models depends on HuggingFace code. It utilizes all methods and functions located in `/splade/hf/` and leverages the full hierarchy of HuggingFace libraries.

- **Indexing and Evaluation**: Indexing and evaluation are performed using a Numba inverted index. This uses the legagy code from SPLADE (all other files outside of `/splade/hf/`).

**Note:** This structure originates from the original code of the Naver repository. Therefore, if you modify the model architecture during training within `/splade/hf/`, ensure you also adjust the model architecture during inference in `/splade/models/` to maintain consistency.

**Configuration**: The SPLADE repository utilizes Hydra for configuration management. You can find and modify the configuration files in `/conf/`.

### Key Scripts

- **`hf_train.py`**: Script to train the model.
- **`index.py`**: Script to index the collection.
- **`retrieve.py`**: Script to perform retrieval.
- **`evaluate.py`**: Script to evaluate the model.
