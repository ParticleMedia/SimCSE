from simcse.models import BertForCL, RobertaForCL
import torch
import sys
from sklearn.metrics.pairwise import cosine_similarity
from train import ModelArguments, DataTrainingArguments, OurTrainingArguments
import os

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
)


class SimCseInference(object):

    def __init__(self, model_args, data_args):
        self.model_args = model_args
        self.max_length = data_args.max_seq_length
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        assert model_args.model_name_or_path
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

        tokenizer_kwargs = {
            "cache_dir": model_args.cache_dir,
            "use_fast": model_args.use_fast_tokenizer,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }
        if model_args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        elif model_args.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        if 'roberta' in model_args.model_name_or_path:
            self.model = RobertaForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
        elif 'bert' in model_args.model_name_or_path:
            self.model = BertForCL.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                model_args=model_args
            )
        else:
            raise NotImplementedError
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model = self.model.to(self.device)

    def calculate_emb(self, sentences: list, return_numpy=True):
        with torch.no_grad():
            inputs = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, sent_emb=True)
            emb = outputs.pooler_output
            if return_numpy:
                emb = emb.numpy()
            return emb

    def similarity_debug(self, sents_0, sents_1):
        embs_0 = self.calculate_emb(sents_0, return_numpy=True)  # suppose N queries
        embs_1 = self.calculate_emb(sents_1, return_numpy=True)  # suppose N queries

        similarities = cosine_similarity(embs_0, embs_1)
        return similarities

    def similarity(self, sents_0, sents_1):
        embs = self.calculate_emb([sents_0] + [sents_1], return_numpy=True)  # suppose N queries
        similarities = cosine_similarity([embs[0]], [embs[1]])
        return similarities


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = SimCseInference(model_args=model_args, data_args=data_args)

    while True:
        print("Input a sentence 0:")
        sent_0 = input()
        print("Input a sentence 1:")
        sent_1 = input()
        similarity_0 = model.similarity(sent_0, sent_1)
        print(f"{similarity_0}\t{sent_0}\t{sent_1}")
        similarity_1 = model.similarity_debug(sent_0, sent_1)
        print(f"{similarity_1}\t{sent_0}\t{sent_1}")
