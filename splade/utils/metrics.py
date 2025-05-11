from collections import Counter

import pytrec_eval
from pytrec_eval import RelevanceEvaluator

import numpy as np

def truncate_run(run, k):
    """truncates run file to only contain top-k results for each query"""
    temp_d = {}
    for q_id in run:
        sorted_run = {k: v for k, v in sorted(run[q_id].items(), key=lambda item: item[1], reverse=True)}
        temp_d[q_id] = {k: sorted_run[k] for k in list(sorted_run.keys())[:k]}
    return temp_d


def mrr_k(run, qrel, k, agg=True):
    evaluator = RelevanceEvaluator(qrel, {"recip_rank"})
    truncated = truncate_run(run, k)
    mrr = evaluator.evaluate(truncated)
    if agg:
        print(len(mrr), len(truncated))
        mrr = sum([d["recip_rank"] for d in mrr.values()]) / max(1, len(mrr))
    return mrr

def judged_k(run, qrel, k, agg=True):
    # evaluator = RelevanceEvaluator(qrel, {"recip_rank"})
    truncated = truncate_run(run, k)

    judged = {}
    i=0
    for qid, ranking in truncated.items():
        if qid not in qrel:
            continue
        i+=1
        judged_q = 0
        for did in ranking:
            if did in qrel[qid]:
                judged_q+=1
        judged[qid] = judged_q
    # print(i, len(truncated))
    
    if agg:
        judged = sum([judged_q for qid, judged_q in judged.items()]) / max(1, len(judged))
    return judged


def evaluate(run, qrel, metric, agg=True, select=None):
    # print(pytrec_eval.supported_measures)
    assert metric in pytrec_eval.supported_measures, print("provide valid pytrec_eval metric")
    evaluator = RelevanceEvaluator(qrel, {metric})
    out_eval = evaluator.evaluate(run)
    res = Counter({})
    if agg:
        for d in out_eval.values():  # when there are several results provided (e.g. several cut values)
            res += Counter(d)
        res = {k: v / len(out_eval) for k, v in res.items()}
        if select is not None:
            string_dict = "{}_{}".format(metric, select)
            if string_dict in res:
                return res[string_dict]
            else:  # If the metric is not on the dict, say that it was 0
                return 0
        else:
            return res
    else:
        return out_eval


def init_eval(metric):
    if metric not in ["MRR@10", "recall@10", "recall@50", "recall@100", "recall@200", "recall@500", "recall@1000"]:
        raise NotImplementedError("provide valid metric")
    if metric == "MRR@10":
        return lambda x, y: mrr_k(x, y, k=10, agg=True)
    else:
        return lambda x, y: evaluate(x, y, metric="recall", agg=True, select=metric.split('@')[1])


def ndcgcut(run_obj, qrel, k):
    res = {}
    for qid in run_obj:
        if qid not in qrel:
            continue
        sorted_run_obj={}
        sorted_run_obj[qid] = {k: v for k, v in sorted(run_obj[qid].items(), key=lambda item: item[1], reverse=True)}
        s = [(qrel[qid][i] if i in qrel[qid] else 0) for i,v in list(sorted_run_obj[qid].items())[:k]]
        ideal_qrel_v = [y for x, y in list(sorted(qrel[qid].items(), key=lambda item: item[1], reverse=True))][:k]
        m = sum([v>0 for v in qrel[qid].values()])
        if m == 0:
            res[qid] = 0
            continue
        dcg = 0
        idcg = 0
        for i in range(k):
            dcg += s[i] / np.log2(i+2)
        for i in range(min(k,m)):
            idcg += ideal_qrel_v[i] / np.log2(i+2)
        res[qid] = dcg/idcg
    agg = 0
    for qid in res:
        agg += res[qid]
    agg /= len(res)
    var = 0
    for qid in res:
        var += (res[qid] - agg)**2
    std = var**0.5 / len(res)
    return res, agg, std