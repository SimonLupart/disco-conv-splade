import json
import os
import pickle
import time
from collections import defaultdict

import numba
import numpy as np
import torch
from tqdm.auto import tqdm
import faiss

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



class EncodeAnserini(Evaluator):
    """Create anserini docs
    """

    def __init__(self, model, config, dataset_name=None, output_name=None, input_type="document"):
        super().__init__(model, config, restore=True)
        self.input_type = input_type
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if dataset_name is not None else config["out_dir"]
        makedir(self.out_dir)
        self.output_key = "d_rep" if self.input_type == "document" else "q_rep"
        self.arg_key = "d_kwargs" if self.input_type == "document" else "q_kwargs"
        self.output_name = output_name
        if self.output_name is not None:
            self.filename = output_name
        else:
            self.filename = "docs_anserini.jsonl" if self.input_type == "document" else "queries_anserini.tsv"

    def index(self, collection_loader, quantization_factor=2):
        vocab_dict = collection_loader.tokenizer.get_vocab()
        vocab_dict = {v: k for k, v in vocab_dict.items()}
        collection_file = open(os.path.join(self.out_dir, self.filename), "w")
        t0 = time.time()
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader)):
                inputs = {k: v for k, v in batch.items() if k not in {"id", "text"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                batch_rep = self.model(**{self.arg_key: inputs})[self.output_key].cpu().numpy()
                for rep, id_, text in zip(batch_rep, batch["id"], batch["text"]):
                    id_ = id_.item()
                    idx = np.nonzero(rep)
                    # then extract values:
                    data = rep[idx]
                    data = np.rint(data * quantization_factor).astype(int)
                    dict_splade = dict()
                    for id_token, value_token in zip(idx[0], data):
                        if value_token > 0:
                            real_token = vocab_dict[id_token]
                            dict_splade[real_token] = int(value_token)
                    if len(dict_splade.keys()) == 0:
                        print("empty input =>", id_)
                        dict_splade[vocab_dict[998]] = 1
                        # in case of empty doc we fill with "[unused993]" token (just to fill and avoid issues
                        # with anserini), in practice happens just a few times ...
                    if self.input_type == "document":
                        dict_ = dict(id=id_, content=text, vector=dict_splade)
                        json_dict = json.dumps(dict_)
                        collection_file.write(json_dict + "\n")
                    else:
                        string_splade = " ".join(
                            [" ".join([str(real_token)] * freq) for real_token, freq in dict_splade.items()])
                        collection_file.write(str(id_) + "\t" + string_splade + "\n")


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



class RerankEvaluator(Evaluator):

    def __init__(self, model, config, dataset_name=None, **kwargs):
        super().__init__(model, config, **kwargs)
        self.init_(config=config,dataset_name=dataset_name)

    def init_(self, config,dataset_name):
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if dataset_name is not None else config["out_dir"]

    def evaluate(self, data_loader, out_dir, reranker_type, model_name="unicamp-dl/mt5-13b-mmarco-100k"):
        makedir(out_dir)
        temp_d = defaultdict(dict)
        logs = open(os.path.join(out_dir, "eval_logs.txt"), "w")
        logs.write("begin evaluation on {} batches ...\n".format(len(data_loader)))
        t0 = time.time()
        with torch.no_grad():  # the model has already been put in eval mode at init
            if reranker_type == "monoT5" or reranker_type == "duoT5":
                if reranker_type == "duoT5":
                    model = DuoT5(model=self.model)
                else:
                    model = MonoT5(model=self.model,use_amp=False,pretrained_model_name_or_path=model_name)
#                    model = MonoT5(model=self.model,use_amp=False,pretrained_model_name_or_path="unicamp-dl/mt5-13b-mmarco-100k")
#                    model = MonoT5(model=self.model,use_amp=False,pretrained_model_name_or_path="castorini/monot5-3b-msmarco-10k")

                for query_id, all_list in tqdm(data_loader):
                    query = all_list[0]
                    texts = all_list[1:]
                    reranked = model.rerank(query, texts)
                    for doc in reranked:
                        temp_d[str(query_id)][str(doc.metadata["docid"])] = doc.score
            else:
                for i, batch in enumerate(tqdm(data_loader)):
                    for k, v in batch.items():
                        if k not in ["q_id", "d_id"]:
                            batch[k] = v.to(self.device)
                    logits = self.model(**{k: v for k, v in batch.items() if k not in {"q_id", "d_id","labels_attention"}})
                    logits = logits[0][:, 0]

                    for q_id, d_id, s in zip(batch["q_id"],
                                            batch["d_id"],
                                            to_list(logits),
                                            ):
                        temp_d[str(q_id)][str(d_id)] = s
        with open(os.path.join(self.out_dir, "run.json"), "w") as handler:
            json.dump(temp_d, handler)
        logs.write("done\ntook about {} hours".format((time.time() - t0) / 3600))
        return temp_d

class PairwisePromptEvaluator(Evaluator):

    def __init__(self, model, config, position_dict=None, dataset_name=None, **kwargs):
        super().__init__(model, config, **kwargs)
        self.init_(config=config, position_dict=position_dict, dataset_name=dataset_name)

    def init_(self, config,dataset_name,position_dict):
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if dataset_name is not None else config["out_dir"]
        self.position_dict = position_dict

    @staticmethod
    def compute_score(results, position_dict):
        scores = defaultdict(dict)
        for qid, did_dict in results.items():
            for did, did_values in did_dict.items():
                scores[qid][did] = 0
                for did2, value in did_values.items():
                    if value > results[qid][did2][did]:
                        scores[qid][did] += 1
                    elif value == results[qid][did2][did]:
                        scores[qid][did] += 0.5
                scores[qid][did] += 0.001/position_dict[qid][did] # for solving ties
        return scores

    def evaluate(self, data_loader, out_dir, reranker_type="PairwisePrompt"):
        makedir(out_dir)
        t0 = time.time()
        with torch.no_grad():  # the model has already been put in eval mode at init
            results = defaultdict(dict)
            for i, batch in enumerate(tqdm(data_loader)):
                for k, v in batch.items():
                    if k not in ["q_id", "d_id_1", "d_id_2"]:
                        batch[k] = v.to(self.device)
                outputs = self.model.generate(batch["input_ids"],max_new_tokens=1,output_scores=True,return_dict_in_generate=True).scores[0]
                result = outputs[:,[71,272]] # 71 is A, 272 is B, hardcoded for FlanT5
                all_scores = torch.nn.functional.softmax(result,dim=-1)

                for qid, did1, did2, score in zip(batch["q_id"],
                                        batch["d_id_1"], batch["d_id_2"],
                                        to_list(all_scores)):
                    if did1 not in results[qid]:
                        results[qid][did1] = defaultdict(float)
                    if did2 not in results[qid]:
                        results[qid][did2] = defaultdict(float)
                    
                    if score[0] > score[1]:
                        results[qid][did1][did2] = 1
                    else:
                        results[qid][did1][did2] = -1    
                    
                    
                    
        temp_d = PairwisePromptEvaluator.compute_score(results, self.position_dict)
        with open(os.path.join(self.out_dir, "run.json"), "w") as handler:
            json.dump(temp_d, handler)
        print("done\ntook about {} hours".format((time.time() - t0) / 3600))
        return temp_d


class DenseIndexing(Evaluator):
    """index dense collection with FAISS
    """

    def __init__(self, model, config, **kwargs):
        super().__init__(model, config, **kwargs)
        self.output_dim = self.model.module.transformer_rep.transformer.config.hidden_size if hasattr(self.model,
                                                                                                      "module") \
            else self.model.transformer_rep.transformer.config.hidden_size  # needed when using DataParallel
        self.index_dir = config["index_dir"] if config is not None else None
        # print(self.output_dim)
        self.cpu_index = faiss.index_factory(self.output_dim, "IDMap,Flat", faiss.METRIC_INNER_PRODUCT)
        # exact brute force index (IndexFlatIP)
        # we use the index factory (with IDMap), because IndexFlatIP does not support adding with ids
        # see: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#then-flat

    def index(self, collection_loader, id_dict=False):
        if self.index_dir is not None:
            makedir(self.index_dir)
            logs = open(os.path.join(self.index_dir, "logs.txt"), "w")
        t0 = time.time()
        # n = len(collection_loader)
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader)):
                # if t % 200 == 0:
                #     print("batch {} / {}".format(t, n), flush=True)
                inputs = {k: v for k, v in batch.items() if k not in {"id"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                batch_documents = self.model(d_kwargs=inputs)["d_rep"]
                batch_ids = to_list(batch["id"])
#                if id_dict:
#                    batch_ids = [str(id_dict[local_id]) for local_id in batch_ids] ## np.ascontiguousarray(
                # print(batch_documents.shape)
                self.cpu_index.add_with_ids(np.ascontiguousarray(batch_documents.cpu()),
                                            np.ascontiguousarray(batch_ids))
                # NOTE: we add with ids, for full MS MARCO collection it's the same (doc id == row id), but not in the
                # general case
        if self.index_dir is not None:
            # save FAISS index => https://github.com/facebookresearch/faiss/issues/421
            faiss.write_index(self.cpu_index, os.path.join(self.index_dir, "faiss_index.FAISS"))
            pickle.dump(id_dict, open(os.path.join(self.index_dir, "doc_dict.pkl"),"wb"))
            logs.write("took about {} hours".format((time.time() - t0) / 3600))
            logs.write("\nindex for model: {}".format(self.config["checkpoint_dir"]))
            logs.write("\nindex contains {} entries".format(self.cpu_index.ntotal))
        else:
            return self.cpu_index


class DenseRetrieval(Evaluator):

    def __init__(self, model, config, dataset_name=None,  index=None, is_beir=False, compute_stats=False, **kwargs):
        super().__init__(model, config, **kwargs)
        assert ("index_dir" in config and index is None) or ("index_dir" not in config and index is not None)
        if "index_dir" in config:
            index_cpu = faiss.read_index(os.path.join(config["index_dir"], "faiss_index.FAISS"))
            self.cpu_index = faiss.index_cpu_to_all_gpus(index_cpu)
            # index1_gpu = faiss.index_cpu_to_gpu_multiple_py(resources, index1)
            print("Read Faiss Index done")
            self.id_dict = pickle.load(open(os.path.join(config["index_dir"], "doc_dict.pkl"),"rb"))
        else:
            self.cpu_index = index
            self.id_dict = None
        self.set_dataset_name(dataset_name)

    def set_dataset_name(self,name):
        self.out_dir = os.path.join(self.config["out_dir"], name) if name is not None else self.config["out_dir"]

    def retrieve(self, q_loader, top_k, name=None, return_d=False, id_dict=False):
        makedir(self.out_dir)
        # logs = open(os.path.join(self.out_dir, "logs.txt"), "w")
        # t0 = time.time()
        # n = len(q_loader)
        res = {}
        with torch.no_grad():
            for t, batch in enumerate(tqdm(q_loader)):
                # if t % 2 == 0:
                #     print("batch {} / {}".format(t, n), flush=True)
                inputs = {k: v for k, v in batch.items() if k not in {"id"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                batch_queries = self.model(q_kwargs=inputs)["q_rep"]
                D, I = self.cpu_index.search(x=np.ascontiguousarray(batch_queries.cpu()), k=top_k) # .cpu()
                for q_id, scores, q_neighbors in zip(to_list(batch["id"]), D, I):
                    # print(q_id)
                    if id_dict:
                        q_id = id_dict[q_id]
                    if self.id_dict:
                        res[str(q_id)] = {str(self.id_dict[key]): float(value) for key, value in zip(q_neighbors, scores)}
                    else:
                        res[str(q_id)] = {str(key): float(value) for key, value in zip(q_neighbors, scores)}
        with open(os.path.join(self.out_dir, "run{}.json".format("_iter_{}".format(name) if name is not None else "")),
                  "w") as handler:
            json.dump(res, handler)
        if return_d:
            return {"retrieval": res}
        # print("done\ntook about {} hours".format((time.time() - t0) / 3600))
        # logs.write("done\ntook about {} hours".format((time.time() - t0) / 3600))
