import json

path_data="DATA/topiocqa_topics/"

# train set TOPIOCQA, index is row number
qrels = {}
with open(path_data+"raw_train.json", "r") as f:
    topiocqa_conv = json.load(f)
    with open(path_data+"queries_rowid_train_last.tsv", "w") as tsv_queries_last:
        with open(path_data+"queries_rowid_train_all.tsv", "w") as tsv_queries_all:
            for i, topic_turn in enumerate(topiocqa_conv):
                conv_turn_id = str(topic_turn["conv_id"])+"_"+str(topic_turn["turn_id"])
                user_question = topic_turn["question"].split("[SEP]")[-1].strip()

                all_user_question = topic_turn["question"].split("[SEP]")
                all_user_question = [t.strip() for t in all_user_question]
                all_user_question.reverse()
                tsv_queries_last.write(str(i)+"\t"+user_question+"\n")
                tsv_queries_all.write(str(i)+"\t"+" [SEP] ".join(all_user_question)+"\n")
                qrels[str(i)] = {topic_turn["positive_ctxs"][0]["passage_id"]: 1}
json.dump(qrels, open(path_data+"qrel_rowid_train.json", "w"))

# test set TOPIOCQA, index is row number
with open(path_data+"raw_dev.json", "r") as f:
    topiocqa_conv = json.load(f)
    with open(path_data+"queries_rowid_dev_last.tsv", "w") as tsv_queries_last:
        with open(path_data+"queries_rowid_dev_all.tsv", "w") as tsv_queries_all:
            for i, topic_turn in enumerate(topiocqa_conv):
                conv_turn_id = str(topic_turn["conv_id"])+"_"+str(topic_turn["turn_id"])
                user_question = topic_turn["question"].split("[SEP]")[-1].strip()

                all_user_question = topic_turn["question"].split("[SEP]")
                all_user_question.reverse()
                tsv_queries_last.write(str(i)+"\t"+user_question+"\n")
                tsv_queries_all.write(str(i)+"\t"+" [SEP] ".join(all_user_question)+"\n")
                qrels[str(i)] = {topic_turn["positive_ctxs"][0]["passage_id"]: 1}
        # mapturnid2rowid[turn_id]=str(i)
json.dump(qrels, open(path_data+"qrel_rowid_dev.json", "w"))
