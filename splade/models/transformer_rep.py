from abc import ABC

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

from ..tasks.amp import NullContextManager
from ..utils.utils import generate_bow, clean_bow, normalize, pruning

"""
we provide abstraction classes from which we can easily derive representation-based models with transformers like SPLADE
with various options (one or two encoders, freezing one encoder etc.) 
"""

class TransformerRep(torch.nn.Module):

    def __init__(self, model_type_or_dir, output, fp16=False):
        """
        output indicates which representation(s) to output from transformer ("MLM" for MLM model)
        model_type_or_dir is either the name of a pre-trained model (e.g. bert-base-uncased), or the path to
        directory containing model weights, vocab etc.
        """
        super().__init__()
        assert output in ("mean", "cls", "hidden_states", "MLM"), "provide valid output"

        # Utiliser un modèle standard (BERT, etc.)
        model_class = AutoModel if output != "MLM" else AutoModelForMaskedLM
        self.transformer = model_class.from_pretrained(model_type_or_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir)
        self.output = output
        self.fp16 = fp16

    def forward(self, **tokens):
        with torch.cuda.amp.autocast() if self.fp16 else NullContextManager():
            # Utilisation d'un modèle standard
            out = self.transformer(**tokens)
            if self.output == "MLM":
                return out
            hidden_states = out[0]  # first element is hidden states, shape (bs, seq_len, hidden_dim)
            if self.output == "mean":
                return torch.sum(hidden_states * tokens["attention_mask"].unsqueeze(-1),
                                    dim=1) / torch.sum(tokens["attention_mask"], dim=-1, keepdim=True)
            elif self.output == "cls":
                return hidden_states[:, 0, :]  # returns [CLS] representation
            else:
                return hidden_states, tokens["attention_mask"]


class SiameseBase(torch.nn.Module, ABC):

    def __init__(self, model_type_or_dir, output, match="dot_product", model_type_or_dir_q=None, freeze_d_model=False,
                 fp16=False, **kwargs):
        super().__init__()
        self.output = output
        assert match in ("dot_product", "cosine_sim"), "specify right match argument"
        self.cosine = True if match == "cosine_sim" else False
        self.match = match
        self.fp16 = fp16
        self.transformer_rep = TransformerRep(model_type_or_dir, output, fp16)
        self.transformer_rep_q = TransformerRep(model_type_or_dir_q,
                                                output, fp16) if model_type_or_dir_q is not None else None
        assert not (freeze_d_model and model_type_or_dir_q is None)
        # print("######@@@@@@@########", freeze_d_model)
        # print("######@@######@@", "######@@@@@@@########", freeze_d_model, flush=True)
        self.freeze_d_model = freeze_d_model
        if freeze_d_model:
            self.transformer_rep.requires_grad_(False)

    def encode(self, kwargs, is_q):
        raise NotImplementedError

    def encode_(self, tokens, is_q=False):
        transformer = self.transformer_rep
        if is_q and self.transformer_rep_q is not None:
            transformer = self.transformer_rep_q
        return transformer(**tokens)

    def train(self, mode=True):
        if self.transformer_rep_q is None:  # only one model, life is simple
            self.transformer_rep.train(mode)
        else:  # possibly freeze d model
            self.transformer_rep_q.train(mode)
            mode_d = False if not mode else not self.freeze_d_model
            self.transformer_rep.train(mode_d)

    def forward(self, **kwargs):
        """forward takes as inputs 1 or 2 dict
        "d_kwargs" => contains all inputs for document encoding
        "q_kwargs" => contains all inputs for query encoding ([OPTIONAL], e.g. for indexing)
        """
        with torch.cuda.amp.autocast() if self.fp16 else NullContextManager():
            out = {}
            do_d, do_q = "d_kwargs" in kwargs, "q_kwargs" in kwargs
            if do_d:
                d_rep = self.encode(kwargs["d_kwargs"], is_q=False)
                if self.cosine:  # normalize embeddings
                    d_rep = normalize(d_rep)
                out.update({"d_rep": d_rep})
            if do_q:
                q_rep = self.encode(kwargs["q_kwargs"], is_q=True)
                if self.cosine:  # normalize embeddings
                    q_rep = normalize(q_rep)
                out.update({"q_rep": q_rep})
            if do_d and do_q:
                if "nb_negatives" in kwargs:
                    # in the cas of negative scoring, where there are several negatives per query
                    bs = q_rep.shape[0]
                    d_rep = d_rep.reshape(bs, kwargs["nb_negatives"], -1)  # shape (bs, nb_neg, out_dim)
                    q_rep = q_rep.unsqueeze(1)  # shape (bs, 1, out_dim)
                    score = torch.sum(q_rep * d_rep, dim=-1)  # shape (bs, nb_neg)
                else:
                    if "score_batch" in kwargs:
                        score = torch.matmul(q_rep, d_rep.t())  # shape (bs_q, bs_d)
                    else:
                        score = torch.sum(q_rep * d_rep, dim=1, keepdim=True)  # shape (bs, )
                out.update({"score": score})
        # for k in out:
        #     print(k+ " size: ", out[k].size())
        return out


class Splade(SiameseBase):
    """SPLADE model
    """

    def __init__(self, model_type_or_dir, model_type_or_dir_q=None, freeze_d_model=False, agg="max", fp16=True):
        super().__init__(model_type_or_dir=model_type_or_dir,
                         output="MLM",
                         match="dot_product",
                         model_type_or_dir_q=model_type_or_dir_q,
                         freeze_d_model=freeze_d_model,
                         fp16=fp16)
        # print("######@@@@@@@########", freeze_d_model)
        # print("######@@@@@@@########", freeze_d_model, flush=True)
        self.output_dim = self.transformer_rep.transformer.config.vocab_size  # output dim = vocab size = 30522 for BERT
        assert agg in ("sum", "max")
        self.agg = agg

    def encode(self, tokens, is_q):
        out = self.encode_(tokens, is_q)["logits"]  # shape (bs, pad_len, voc_size)
        if self.agg == "sum":
            return torch.sum(torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1), dim=1)
        else:
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1), dim=1)
            return values
            # 0 masking also works with max because all activations are positive


class SpladeDoc(SiameseBase):
    """SPLADE without query encoder
    """

    def __init__(self, model_type_or_dir, model_type_or_dir_q=None,
                 freeze_d_model=False, agg="sum", fp16=True):
        super().__init__(model_type_or_dir=model_type_or_dir,
                         output="MLM",
                         match="dot_product",
                         model_type_or_dir_q=model_type_or_dir_q,
                         freeze_d_model=freeze_d_model,
                         fp16=fp16)
        assert model_type_or_dir_q is None
        assert not freeze_d_model
        self.output_dim = self.transformer_rep.transformer.config.vocab_size
        self.pad_token = self.transformer_rep.tokenizer.special_tokens_map["pad_token"]
        self.cls_token = self.transformer_rep.tokenizer.special_tokens_map["cls_token"]
        self.sep_token = self.transformer_rep.tokenizer.special_tokens_map["sep_token"]
        self.pad_id = self.transformer_rep.tokenizer.vocab[self.pad_token]
        self.cls_id = self.transformer_rep.tokenizer.vocab[self.cls_token]
        self.sep_id = self.transformer_rep.tokenizer.vocab[self.sep_token]
        assert agg in ("sum", "max")
        self.agg = agg

    def encode(self, tokens, is_q):
        if is_q:
            q_bow = generate_bow(tokens["input_ids"], self.output_dim, device=tokens["input_ids"].device)
            q_bow[:, self.pad_id] = 0
            q_bow[:, self.cls_id] = 0
            q_bow[:, self.sep_id] = 0
            # other the pad, cls and sep tokens are in bow
            return q_bow
        else:
            out = self.encode_(tokens)["logits"]  # shape (bs, pad_len, voc_size)
            if self.agg == "sum":
                return torch.sum(torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1), dim=1)
            else:
                values, _ = torch.max(torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1), dim=1)
                return values
                # 0 masking also works with max because all activations are positive


class SpladeTopK(SiameseBase):
    """
    model with topk 
    """

    def __init__(self, model_type_or_dir, model_type_or_dir_q=None,
                 freeze_d_model=False, agg="max",fp16=False, output='MLM', top_d=32, top_q=5, **kwargs):
        super().__init__(model_type_or_dir=model_type_or_dir,
                         output=output,
                         match='dot_product',
                         model_type_or_dir_q=model_type_or_dir_q,
                         freeze_d_model=freeze_d_model,fp16=fp16,
                         **kwargs)
        self.output_dim = self.transformer_rep.transformer.config.vocab_size  # output dim = vocab size
        assert agg in ("sum", "max","cls")
        self.agg = agg
        self.top_d = top_d # If -1 no threshold
        self.top_q = top_q

    def encode(self, tokens, is_q):
        out = self.encode_(tokens, is_q)['logits']  # shape (bs, pad_len, voc_size)
        if self.agg == "sum":
            rep = torch.sum(torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1), dim=1)
        elif self.agg == "cls":
            rep = (torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1))[:, 0, :]
        else:
            values, _ = torch.max(torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1), dim=1)
            rep = values
        top_local = self.top_q if is_q else self.top_d
        if top_local > 0: # If -1 no threshold
            return pruning(rep,top_local,len(rep.size())-1)
        else:
            return rep

class SpladeLexical(SiameseBase):
    """
    document expansion model with weigthed query
    """

    def __init__(self, model_type_or_dir, model_type_or_dir_q=None,
                 freeze_d_model=False, lexical_type="query", agg="sum",fp16=False):
        super().__init__(model_type_or_dir=model_type_or_dir,
                         output='MLM',
                         match='dot_product',
                         model_type_or_dir_q=model_type_or_dir_q,
                         freeze_d_model=freeze_d_model,fp16=fp16)
        self.output_dim = self.transformer_rep.transformer.config.vocab_size
        self.pad_token = self.transformer_rep.tokenizer.special_tokens_map["pad_token"]
        self.cls_token = self.transformer_rep.tokenizer.special_tokens_map["cls_token"]
        self.sep_token = self.transformer_rep.tokenizer.special_tokens_map["sep_token"]
        self.mask_token = self.transformer_rep.tokenizer.special_tokens_map["mask_token"]
        self.pad_id = self.transformer_rep.tokenizer.vocab[self.pad_token]
        self.cls_id = self.transformer_rep.tokenizer.vocab[self.cls_token]
        self.sep_id = self.transformer_rep.tokenizer.vocab[self.sep_token]
        self.mask_id = self.transformer_rep.tokenizer.vocab[self.mask_token]
        assert agg in ("sum", "max","cls")
        self.agg = agg
        self.lexical_type = lexical_type
        assert lexical_type in ("query","document","both")

    def encode(self, tokens, is_q):
        bow = generate_bow(tokens["input_ids"], self.output_dim, device=tokens["input_ids"].device)
        bow = clean_bow(bow, pad_id = self.pad_id, cls_id=self.cls_id, sep_id=self.sep_id, mask_id=self.mask_id)

        out = self.encode_(tokens, is_q)['logits']  # shape (bs, pad_len, voc_size)
        if self.agg == "sum":
            rep = torch.sum(torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1), dim=1)
        elif self.agg == "cls":
            rep = (torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1))[:, 0, :]
        else:
            rep = torch.max(torch.log(1 + torch.relu(out)) * tokens["attention_mask"].unsqueeze(-1), dim=1).values
        
        if self.lexical_type == "both":
            rep = rep * bow
        elif is_q and self.lexical_type == "query": # If lexical is query and is query
            rep = rep * bow
        elif not is_q and self.lexical_type == "document": # If lexical is document and is document
            rep = rep * bow

        return rep

