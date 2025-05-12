import json
import os
import pickle
import time
from collections import defaultdict

import numba
import numpy as np
import torch
from tqdm.auto import tqdm

import time

from ..indexing.inverted_index import IndexDictOfArray
from ..losses.regularization import L0
from ..tasks.base.evaluator import Evaluator
from ..utils.utils import makedir, to_list


class SparseIndexing(Evaluator):
    """sparse indexing
    """

    def __init__(self, model, config, compute_stats=False, dim_voc=None, is_query=False, force_new=True,**kwargs):
        super().__init__(model, config, **kwargs)
        self.index_dir = config["index_dir"] if config is not None else None
        self.sparse_index = IndexDictOfArray(self.index_dir, dim_voc=dim_voc, force_new=force_new)
        self.compute_stats = compute_stats
        self.is_query = is_query
        if self.compute_stats:
            self.l0 = L0()

    def index(self, collection_loader, id_dict=None):
        doc_ids = []
        if self.compute_stats:
            stats = defaultdict(float)
        count = 0
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader)):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id"}}
                if self.is_query:
                    batch_documents = self.model(q_kwargs=inputs)["q_rep"]
                else:
                    batch_documents = self.model(d_kwargs=inputs)["d_rep"]
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_documents).item()
                row, col = torch.nonzero(batch_documents, as_tuple=True)
                data = batch_documents[row, col]
                row = row + count
                batch_ids = to_list(batch["id"])
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                count += len(batch_ids)
                doc_ids.extend(batch_ids)
                self.sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(),
                                                     n_docs=len(batch_ids))
        if self.compute_stats:
            stats = {key: value / len(collection_loader) for key, value in stats.items()}
        if self.index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb"))
            print("done iterating over the corpus...")
            print("index contains {} posting lists".format(len(self.sparse_index)))
            print("index contains {} documents".format(len(doc_ids)))
            if self.compute_stats:
                with open(os.path.join(self.index_dir, "index_stats.json"), "w") as handler:
                    json.dump(stats, handler)
        else:
            # if no index_dir, we do not write the index to disk but return it
            for key in list(self.sparse_index.index_doc_id.keys()):
                # convert to numpy
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key],
                                                                  dtype=np.float32)
            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            if self.compute_stats:
                out["stats"] = stats
            return out


class SparseRetrieval(Evaluator):
    """retrieval from SparseIndexing
    """

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in numba.prange(len(retrieved_indexes)):
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, model, config, dim_voc, dataset_name=None, index_d=None, compute_stats=False, is_beir=False,
                 **kwargs):
        super().__init__(model, config, **kwargs)
        assert ("index_dir" in config and index_d is None) or (
                "index_dir" not in config and index_d is not None)
        if "index_dir" in config:
            self.sparse_index = IndexDictOfArray(config["index_dir"], dim_voc=dim_voc)
            self.doc_ids = pickle.load(open(os.path.join(config["index_dir"], "doc_ids.pkl"), "rb"))
        else:
            self.sparse_index = index_d["index"]
            self.doc_ids = index_d["ids_mapping"]
            for i in range(dim_voc):
                # missing keys (== posting lists), causing issues for retrieval => fill with empty
                if i not in self.sparse_index.index_doc_id:
                    self.sparse_index.index_doc_id[i] = np.array([], dtype=np.int32)
                    self.sparse_index.index_doc_value[i] = np.array([], dtype=np.float32)
        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if (dataset_name is not None and not is_beir) \
            else config["out_dir"]
        self.doc_stats = index_d["stats"] if (index_d is not None and compute_stats) else None
        self.compute_stats = compute_stats
        if self.compute_stats:
            self.l0 = L0()

    def retrieve(self, q_loader, top_k, name=None, return_d=False, id_dict=False, threshold=0, store_rep=False):
        all_rep=[]
        id_rep=[]
        latency_enc = []
        latency_retrieval = []

        makedir(self.out_dir)
        if self.compute_stats:
            makedir(os.path.join(self.out_dir, "stats"))
        res = defaultdict(dict)
        if self.compute_stats:
            stats = defaultdict(float)
        with torch.no_grad():
            for t, batch in enumerate(tqdm(q_loader)):
                # if store_rep and t==1000:
                #     break
                q_id = to_list(batch["id"])[0]
                if id_dict:
                    q_id = id_dict[q_id]
                inputs = {k: v for k, v in batch.items() if k not in {"id"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)

                encoding_start = time.time()
                query = self.model(q_kwargs=inputs)["q_rep"]  # we assume ONE query per batch here
                encoding_end=time.time()

                if self.compute_stats:
                    stats["L0_q"] += self.l0(query).item()
                # TODO: batched version for retrieval
                row, col = torch.nonzero(query, as_tuple=True)
                values = query[to_list(row), to_list(col)]
                if store_rep:
                    all_rep.append([to_list(col),to_list(values)])
                    id_rep.append(q_id)

                col_numba=col.cpu().numpy()
                values_numba=values.cpu().numpy().astype(np.float32)
                size_collection=self.sparse_index.nb_docs()

                retrieval_start=time.time()
                filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                                  self.numba_index_doc_values,
                                                                  col_numba,
                                                                  values_numba,
                                                                  threshold=threshold,
                                                                  size_collection=size_collection)
                retrieval_end = time.time()

                latency_enc.append(encoding_end-encoding_start)
                latency_retrieval.append(retrieval_end-retrieval_start)
                # threshold set to 0 by default, could be better
                filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
                
                for id_, sc in zip(filtered_indexes, scores):
                    res[str(q_id)][str(self.doc_ids[id_])] = float(sc)

        print("Average Latency Encoding:", np.mean(latency_enc), "for {} queries".format(len(latency_enc)))
        print("Average Latency Retrieval:", np.mean(latency_retrieval), "for {} queries".format(len(latency_retrieval)))
        
        if self.compute_stats:
            stats = {key: value / len(q_loader) for key, value in stats.items()}
        if self.compute_stats:
            with open(os.path.join(self.out_dir, "stats",
                                   "q_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                      "w") as handler:
                json.dump(stats, handler)
            if self.doc_stats is not None:
                with open(os.path.join(self.out_dir, "stats",
                                       "d_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                          "w") as handler:
                    json.dump(self.doc_stats, handler)
        with open(os.path.join(self.out_dir, "run{}.json".format("_iter_{}".format(name) if name is not None else "")),
                  "w") as handler:
            json.dump(res, handler)
        if store_rep:
            with open(os.path.join(self.out_dir, "all_rep{}.pkl".format("_iter_{}".format(name) if name is not None else "")),
                    "wb") as handler:
                pickle.dump((id_rep,all_rep), handler)
        if return_d:
            out = {"retrieval": res}
            if self.compute_stats:
                out["stats"] = stats if self.doc_stats is None else {**stats, **self.doc_stats}
            return out

    def batch_retrieve(self, q_loader, top_k, name=None, return_d=False, id_dict=False, threshold=0, store_rep=False):
        all_rep = []  # No changes
        id_rep = []  # No changes

        latency_enc = []
        latency_retrieval = []

        makedir(self.out_dir)  # No changes
        if self.compute_stats:
            makedir(os.path.join(self.out_dir, "stats"))  # No changes
        res = defaultdict(dict)  # No changes
        if self.compute_stats:
            stats = defaultdict(float)  # No changes

        with torch.no_grad():  # No changes
            for t, batch in enumerate(tqdm(q_loader)):
                # === HANDLE MULTIPLE QUERY IDS IN A BATCH ===
                q_ids = to_list(batch["id"])  # Get a list of query IDs (instead of assuming only one ID per batch)
                if id_dict:
                    q_ids = [id_dict[q_id] for q_id in q_ids]  # Map IDs using id_dict, if provided

                inputs = {k: v for k, v in batch.items() if k not in {"id"}}  # No changes
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)  # No changes

                # === PROCESS QUERIES IN BATCH ===
                encoding_start = time.time()
                queries = self.model(q_kwargs=inputs)["q_rep"]  # Compute representations for all queries in the batch
                encoding_end = time.time()

                if self.compute_stats:
                    stats["L0_q"] += self.l0(queries).item()  # Adjusted to process all queries

                # === BATCHED RETRIEVAL ===
                for i, query in enumerate(queries):  # Iterate through each query representation in the batch
                    q_id = q_ids[i]  # Corresponding query ID

                    if query.ndim == 1:  
                        query = query.unsqueeze(0)  # Add a batch dimension if missing

                    # Sparse vector extraction for the current query
                    row, col = torch.nonzero(query, as_tuple=True)
                    values = query[to_list(row), to_list(col)]

                    if store_rep:
                        all_rep.append([to_list(col), to_list(values)])  # Store current query's sparse representation
                        id_rep.append(q_id)  # Store corresponding query ID

                    retrieval_start = time.time()
                    # Perform retrieval for the current query
                    filtered_indexes, scores = self.numba_score_float(
                        self.numba_index_doc_ids,
                        self.numba_index_doc_values,
                        col.cpu().numpy(),
                        values.cpu().numpy().astype(np.float32),
                        threshold=threshold,
                        size_collection=self.sparse_index.nb_docs()
                    )
                    retrieval_end = time.time()

                    # Select top-k results for the current query
                    filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)

                    for id_, sc in zip(filtered_indexes, scores):
                        res[str(q_id)][str(self.doc_ids[id_])] = float(sc)

        # === REMAINING CODE UNCHANGED ===
        if self.compute_stats:
            stats = {key: value / len(q_loader) for key, value in stats.items()}
        if self.compute_stats:
            with open(os.path.join(self.out_dir, "stats",
                                   "q_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                      "w") as handler:
                json.dump(stats, handler)
            if self.doc_stats is not None:
                with open(os.path.join(self.out_dir, "stats",
                                       "d_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                          "w") as handler:
                    json.dump(self.doc_stats, handler)
        with open(os.path.join(self.out_dir, "run{}.json".format("_iter_{}".format(name) if name is not None else "")),
                  "w") as handler:
            json.dump(res, handler)
        if store_rep:
            with open(os.path.join(self.out_dir, "all_rep{}.pkl".format("_iter_{}".format(name) if name is not None else "")),
                      "wb") as handler:
                pickle.dump((id_rep, all_rep), handler)
        if return_d:
            out = {"retrieval": res}
            if self.compute_stats:
                out["stats"] = stats if self.doc_stats is None else {**stats, **self.doc_stats}
            return out


class SparseApproxEvalWrapper(Evaluator):
    """
    wrapper for sparse indexer + retriever during training
    """

    def __init__(self, model, config, collection_loader, q_loader, **kwargs):
        super().__init__(model, config, **kwargs)
        self.collection_loader = collection_loader
        self.q_loader = q_loader
        self.model_output_dim = self.model.module.output_dim if hasattr(self.model, "module") else self.model.output_dim

    def index_and_retrieve(self, i):
        indexer = SparseIndexing(self.model, config=None, restore=False, compute_stats=True)
        sparse_index_d = indexer.index(self.collection_loader)
        retriever = SparseRetrieval(self.model, self.config, dim_voc=self.model_output_dim, index_d=sparse_index_d,
                                    restore=False, compute_stats=True)
        return retriever.retrieve(self.q_loader, top_k=self.config["top_k"], name=i, return_d=True)
