import argparse
import json

from ..utils.metrics import mrr_k, evaluate, judged_k, ndcgcut


def load_and_evaluate(qrel_file_path, run_file_path, metric, agg=True):
    with open(qrel_file_path) as reader:
        qrel = json.load(reader)
    with open(run_file_path) as reader:
        run = json.load(reader)

    # for trec, qrel_binary.json should be used for recall etc., qrel.json for NDCG.
    # if qrel.json is used for binary metrics, the binary 'splits' are not correct
    if "TREC" in qrel_file_path:
        assert ("binary" not in qrel_file_path) == (metric == "ndcg" or metric == "ndcg_cut")
    if metric == "mrr_10":
        res = mrr_k(run, qrel, k=10, agg=agg)
        if agg:
            print("MRR@10:", res)
        return {"mrr_10": res}
    elif metric == "mrr_1000":
        res = mrr_k(run, qrel, k=1000, agg=agg)
        if agg:
            print("MRR@1000:", res)
        return {"mrr_1000": res}
    elif metric == "judged@10":
        res = judged_k(run, qrel, k=10)
        if agg:
            print("judged@10:", res)
        return {"judged@10": res}
    elif metric == "judged@100":
        res = judged_k(run, qrel, k=100)
        if agg:
            print("judged@100:", res)
        return {"judged@100": res}
    elif metric == "judged@1000":
        res = judged_k(run, qrel, k=1000)
        if agg:
            print("judged@1000:", res)
        return {"judged@1000": res}
    elif metric == "ndcg@3":
        if agg:
            res = ndcgcut(run, qrel, k=3)[1] # 0 is not agg, 1 is agg
            print("ndcg@3:", res)
        else:
            res = ndcgcut(run, qrel, k=3)[0]
        return {"ndcg@3": res}
    else:
        res = evaluate(run, qrel, metric=metric, agg=agg)
        if agg:
            print(metric, "==>", res)
        return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrel_file_path")
    parser.add_argument("--run_file_path")
    parser.add_argument("--metric", default="mrr_10")
    args = parser.parse_args()
    load_and_evaluate(args.qrel_file_path, args.run_file_path, args.metric)
