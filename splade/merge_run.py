import json
import sys

# if len(sys.argv)!=4:
#     print("python merge_run.py <run_1> .. <run_n> <out>")
#     exit(1)
runs = []
for i, run in enumerate(sys.argv):
    if i==0:
        continue
    if (i+1)==len(sys.argv):
        output_file = run
        continue
    runs.append(run)
# run1 = sys.argv[1]
# run2 = sys.argv[2]
# out = sys.argv[3]
    
run_merged = {}
for run in runs:
    with open(run, "r") as runf:
        run_obj = json.load(runf)
        for qid in run_obj:
            l = run_obj[qid]
            if qid not in run_merged:
                run_merged[qid] = l
            else:
                run_merged[qid] = {**run_merged[qid], **l}
print("merged")
res = {}
for qid in run_merged:
    if qid not in res:
        res[qid] = {}
    for k,v in sorted(run_merged[qid].items(),key=lambda item: item[1], reverse=True)[:1000]:
        if k not in res[qid]:
            res[qid][k] = v
        if len(res[qid]) >= 1000:
            break
print("cutted")
with open(output_file, "w") as f:
    json.dump(res,f)

# # with open(f"/ivi/ilps/personal/slupart/cosplade_d/eval/runs_ours/{run1}.json", "r") as runf1:
# #     with open(f"/ivi/ilps/personal/slupart/cosplade_d/eval/runs_ours/{run2}.json", "r") as runf2:
#         run_obj1 = json.load(runf1)
#         run_obj2 = json.load(runf2)
#         res = {}
#         count_error = 0
#         for qid in set().union(run_obj1, run_obj2):
#             l1 = run_obj1[qid]
#             l2 = run_obj2[qid]
#             l = {**l1,**l2}
#             res[qid] = dict()
#             for k,v in sorted(l.items(),key=lambda item: item[1], reverse=True)[:3000]:
#                 if k not in res[qid]:
#                     res[qid][k] = v
#                 if len(res[qid]) >= 1000:
#                     break
#             print(len(res[qid]))
#             if len(res[qid]) == 0:
#                 print(qid)
#                 count_error += 1
#         print(f"There are {count_error} empty vectors")
# with open(f"/ivi/ilps/personal/slupart/cosplade_d/eval/runs_ours/{out}.json", "w") as f:
#     json.dump(res,f)