import json
import gzip
import pickle
import os
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import random

import numpy as np


class DatasetPreLoad():
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    """

    def __init__(self, data_dir, id_style, filter=False):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        self.id_style = id_style

        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        self.line_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        print("Preloading dataset", flush=True)
        with open(self.data_dir) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    # if i > 6000000:
                    #     break
                    if filter:
                        id_, text, title = line.split("\t")  # first column is id
                        id_ = id_.strip()
                        data = title + ". " + text # text
                    else:
                        id_, *data = line.split("\t")  # first column is id
                        id_ = id_.strip()
                        data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[i] = data
                        self.line_dict[i] = id_.strip()
                    else:
                        self.data_dict[id_] = data.strip()
                        self.line_dict[id_] = i

        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex


    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]


class L2I_Dataset(Dataset): 
    """ 
    output : (queries, docs), scores
    """

    def __init__(self, document_dir, query_dir, qrels_path, n_negatives=2, nqueries=-1,training_file_path=None, training_data_type=None, filter_data=None, topk=(0,1000), norm=False):

        print("START DATASET: %s"%document_dir, flush=True)
        self.document_dataset = DatasetPreLoad(document_dir,id_style="content_id",filter=("topiocqa" in document_dir))
        self.query_dataset = DatasetPreLoad(query_dir,id_style="content_id")

        print(topk[0], topk[1])
        
        self.samples = dict()
        if qrels_path is not None:
            print("Loading qrel: %s"%qrels_path, flush=True)
            with open(qrels_path) as reader:
                self.qrels = json.load(reader)
                
            # select a subset of  queries 
            if nqueries > 0:
                from itertools import islice
                self.qrels = dict(islice(self.qrels.items(), nqueries))
        
            ### mapping to str ids   ###
            self.qrels = {str(k):{str(k2):v2 for k2,v2 in v.items()} for k,v in self.qrels.items()  }
            ### filtering non positives
            self.qrels={k:{k2:v2 for k2,v2 in v.items() if int(v2)>=1} for k,v in self.qrels.items() }
        else:
            self.qrels = None

        self.samples = dict()        
        self.n_negatives = n_negatives

        print("READING TRAINING FILE (%s)"%training_data_type, flush=True)
        if training_data_type == 'saved_pkl':
            # output of the "others" (filter already done)
            # data_type: saved_pkl
            # format: [POS:[NEG:score,NEG:score,...]] 
            self.samples = pickle.load(open(training_file_path,"rb"))

        elif training_data_type == 'pkl_dict':
            # a la Nils
            # data_type: pkl_dict
            # cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz: qid/did are int!
            # with gzip.open(training_file_path, 'rb') as fIn:
            with open(training_file_path, 'rb') as fIn:
                self.samples = pickle.load(fIn)
                # cast int into str for qids/dids
                # and filter out ig not enough negatives 
                self.samples = {str(k):{str(k2):float(v2) for k2,v2 in v.items()} for k,v in self.samples.items() if ( str(k) in self.qrels and len(v.keys()) >  len(self.qrels[str(k)]) )}

                  
        elif training_data_type == 'trec':
            #data_type:trec
            with open(training_file_path) as reader:
                for line in tqdm(reader):
                    qid, _, did, _, score, _ = line.split(" ")
                    # if query subset used
                    if self.qrels is None  or str(qid) in self.qrels:
                        if str(qid) not in self.samples:
                            self.samples[str(qid)] = dict()
                        self.samples[str(qid)][str(did)]=float(score)

        elif training_data_type == 'json':
            # l2i output
            #data_type: json
            with open(training_file_path) as reader:
                self.result_path = json.load(reader)
            for qid, documents in tqdm(self.result_path.items()):
                # if query subset used
                if  self.qrels is None or str(qid) in self.qrels:
                    if str(qid) not in self.samples:
                        self.samples[str(qid)] = dict()
                    for did, score in sorted(documents.items(), reverse=True, key=lambda item: item[1])[topk[0]:][:topk[1]]:
                        self.samples[str(qid)][str(did)]=float(score)
        elif training_data_type == 'json_filter':
            if norm:
                USE_NORM=True
                norm_dict = json.load(open(norm, "r"))
            else:
                USE_NORM=False
            print("USE_NORM:", USE_NORM)
            # self.qrels contains {topic_turn_id:{topic_turn_id: pos_passage_answer}}
            # self.qrels = json.load(open("/gpfs/work4/0/prjs0871/ultrachat/from_ivi/qrels.json", "r"))
            # filter_neg = json.load(open("/gpfs/work4/0/prjs0871/qrecc/filter_collection/filter.json", "r"))
            if filter_data:
                filter_neg = json.load(open(filter_data, "r"))
            with open(training_file_path) as reader:
                self.result_path = json.load(reader)
            for qid, documents in tqdm(self.result_path.items()):
                if qid in ["92_5", "670_3", "7760_1"]:
                    continue

                # if query subset used
                # self.qrels[str(qid)] = {}
                if self.qrels is None or str(qid) in self.qrels:
                    if USE_NORM:
                        if filter_data:
                            teacher_scores = np.array([s for did, s in documents.items() if did in filter_neg[str(qid)]])
                            students_scores = np.array([s for did, s in norm_dict[str(qid)].items() if did in filter_neg[str(qid)]])
                        else:
                            teacher_scores = np.array([s for did, s in documents.items()])
                            students_scores = np.array([s for did, s in norm_dict[str(qid)].items()])

                        max_teacher = float(np.max(teacher_scores))
                        min_teacher = float(np.min(teacher_scores))
                        max_student = float(np.max(students_scores))
                        min_student = float(np.min(students_scores))
                        # Avoid division by zero by adding a small epsilon
                        epsilon = 1e-8

                        if str(qid) not in self.samples:
                            self.samples[str(qid)] = dict()
                        for did, score in sorted(documents.items(), reverse=True, key=lambda item: item[1]):
                            if not filter_data:
                                self.samples[str(qid)][str(did)]= float(((float(score) - min_teacher) / (max_teacher - min_teacher + epsilon)) * (max_student - min_student + epsilon) + min_student)
                            else:
                                if str(did) in filter_neg[str(qid)]:
                                    # Normalize the scores
                                    self.samples[str(qid)][str(did)]= float(((float(score) - min_teacher) / (max_teacher - min_teacher + epsilon)) * (max_student - min_student + epsilon) + min_student)
                        if len(self.samples)<10:
                            print(float(((float(score) - min_teacher) / (max_teacher - min_teacher + epsilon)) * (max_student - min_student + epsilon) + min_student))
                            print(float(score), max_teacher,min_teacher,max_student,min_student,len(teacher_scores),len(students_scores) )


                        
                    else:
                        if str(qid) not in self.samples:
                            self.samples[str(qid)] = dict()
                        for did, score in sorted(documents.items(), reverse=True, key=lambda item: item[1]):
                            # if len(filter_neg[str(qid)]) != 32:
                            #     print(len(filter_neg[str(qid)]))
                            # print(filter_neg[str(qid)])
                            if str(did) in filter_neg[str(qid)]:
                                # if str(did) == "clueweb22-en0025-83-04467:13":
                                #     print(str(qid), filter_neg[str(qid)])
                                self.samples[str(qid)][str(did)]=float(score)
        elif training_data_type == 'json_with_explicit_pos':
            # self.qrels contains {topic_turn_id:{topic_turn_id: pos_passage_answer}}
            self.qrels = json.load(open("/gpfs/work4/0/prjs0871/ultrachat/from_ivi/qrels.json", "r"))
            # filter_neg = json.load(open("/gpfs/work4/0/prjs0871/ultrachat/from_ivi/filter.json", "r"))
            filter_neg = json.load(open(filter_data, "r"))
            with open(training_file_path) as reader:
                self.result_path = json.load(reader)
            for qid, documents in tqdm(self.result_path.items()):
                # if query subset used
                # self.qrels[str(qid)] = {}
                if  self.qrels is None or str(qid) in self.qrels:
                    if str(qid) not in self.samples:
                        self.samples[str(qid)] = dict()
                    for did, score in sorted(documents.items(), reverse=True, key=lambda item: item[1]):
                        # if len(filter_neg[str(qid)]) != 32:
                        #     print(len(filter_neg[str(qid)]))
                        # print(filter_neg[str(qid)])
                        if str(did) in filter_neg[str(qid)]:
                            # if str(did) == "clueweb22-en0025-83-04467:13":
                            #     print(str(qid), filter_neg[str(qid)])
                            self.samples[str(qid)][str(did)]=float(score)

            # self.qrels =
        else:
            raise NotImplementedError('training_data_type must be in [saved_pkl, pkl_dict, trec, json]')
        self.training_data_type = training_data_type


        self.query_list = list(self.samples.keys())
        print("QUERY SIZE = ", len(self.query_list))
        assert  len(self.query_list) > 0 
       

    def __len__(self):
        return len(self.query_list)


    def __getitem__(self, idx):
        query = self.query_list[idx]
        q = self.query_dataset[query][1]
        positives = list(self.qrels[query].keys())
        
        candidates = [x for x in self.samples[query] if x not in positives]
        # print(len(candidates))

        if len(candidates) <= self.n_negatives:
            negative_ids = random.choices(candidates,k=self.n_negatives)
        else:
            negative_ids = random.sample(candidates,self.n_negatives)

        if self.training_data_type == 'json_with_explicit_pos':
            d_pos = self.qrels[query][query]
        else:
            positive = random.sample(positives,1)[0]
            d_pos = self.document_dataset[positive][1]
        negatives = [self.document_dataset[negative][1].strip() for negative in negative_ids]
        q = q.strip()
        d_pos = d_pos.strip()
        
        docs = [q,d_pos]
        docs.extend(negatives)
        scores_negatives = [self.samples[query][negative] for negative in negative_ids]
        try: # If there's a score for the positive on the file it uses that score
            scores = [self.samples[query][positive]]
        except: # KeyError: # else it uses the best score of the negatives.
            scores = [max(v for k,v in self.samples[query].items())]
        scores.extend(scores_negatives)
        scores = torch.tensor(scores)
        scores = scores.view(1,-1)
        return docs, scores


class TRIPLET_Dataset():
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.data_dict = {}  # => dict that maps the id to the line offset (position of pointer in the file)
        print("READING TRANING FILE (triplet): %s"%(data_dir), flush=True)
        with open(self.data_dir) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    query, pos, neg = line.split("\t")  # first column is id
                    docs = [query.strip(), pos.strip(), neg.strip()]
                    scores = torch.tensor([0, 0, 0])
                    scores = scores.view(1,-1)
                    self.data_dict[i] = docs, scores
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        return self.data_dict[idx]

