from simcse.model_inference import SimCseInference
import sys
from train import ModelArguments, DataTrainingArguments, OurTrainingArguments
import os
from simple_celery.huggingface_util.inference.batch_inference_by_redis.do_inference_redis_text import start_to_do_inference_redis
from simple_celery.utils.argparser_util import parse_string_args
import argparse
from transformers import (
    HfArgumentParser,
)


def get_model_call(model):

    def model_call(inputs):
        return model.calculate_emb(inputs[0], return_numpy=False)

    return model_call


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, remaining_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    parser_1 = argparse.ArgumentParser()
    parser_1.add_argument("-o", "--redis_host", type=str, default='127.0.0.1')
    parser_1.add_argument("-p", "--redis_port", type=int, default=6379)
    parser_1.add_argument('-q', "--redis_q", type=str, default='simcse_inference')
    remaining_args = parse_string_args(parser_1, " ".join(remaining_args))

    model = SimCseInference(model_args=model_args, data_args=data_args)

    start_to_do_inference_redis(remaining_args.redis_host, remaining_args.redis_port, remaining_args.redis_q,
                                OurTrainingArguments.per_device_eval_batch_size,
                                get_model_call(model))
