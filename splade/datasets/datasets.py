import gzip
import json
import os
import pickle
import random

from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CollectionDatasetPreLoad(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir, id_style, max_sample=None, topiocqa=False):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        # id_style indicates how we access the doc/q (row id or doc/q id)
        self.id_style = id_style
        self.data_dict = {}
        self.line_dict = {}
        print("Preloading dataset")
        curr_id = 0
        if ".tsv" not in self.data_dir:
            path_collection = os.path.join(self.data_dir, "raw.tsv")
        else:
            path_collection = self.data_dir
        with open(path_collection) as reader:
            for i, line in enumerate(tqdm(reader)):
                if max_sample and i>max_sample:
                    break
                if len(line) > 1:
                    if topiocqa:
                        if i==0: # header
                            continue
                        id_, text, title = line.split("\t")  # first column is id
                        id_ = id_.strip()
                        data = title+". "+text # text
                    else:
                        if len(line.split("\t")) != 2:
                            continue
                        id_, *data = line.split("\t")  # first column is id
                        data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[curr_id] = data
                        self.line_dict[curr_id] = id_.strip()
                        curr_id+=1
                    else:
                        self.data_dict[id_] = data.strip()
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]
