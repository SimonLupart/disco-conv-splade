from transformers import DefaultDataCollator
import torch

def flatten(xss):
    return [x for xs in xss for x in xs]


class L2I_Collator(DefaultDataCollator):
    def __init__(self, tokenizer, max_length=350, *args, **kwargs):
        super(L2I_Collator, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def torch_call(self, examples):
        docs,scores = zip(*examples)
        docs = flatten(docs)
        scores = torch.cat(scores,dim=0)
        tokenized = self.tokenizer(docs,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")       
        tokenized["scores"] = scores
        return tokenized

