from typing import List

import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator, EvalPrediction
from datasets import Dataset
import transformers
import numpy as np

from bertserini.utils.utils_qa import postprocess_qa_predictions
from bertserini.reader.base import Reader, Question, Context, Answer
from bertserini.reader.fid_model import FiDT5
from bertserini.reader.fid_data import FidDataset, Collator

from datasets.utils import logging

# Note: to run fid: you must have Transformers version <= 4.10, and then mute the following error
#  File "/home/y247xie/miniconda3/envs/bertserini/lib/python3.8/site-packages/transformers/generation_utils.py", line 917, in generate
#     raise ValueError("Make sure that `model_kwargs` include `encoder_outputs` of type `ModelOutput`.")

__all__ = ['BERT']

transformers.utils.logging.set_verbosity_error()

class FiD(Reader):
    def __init__(self, args):
        self.model_args = args
        if self.model_args.tokenizer_name is None:
            self.model_args.tokenizer_name = self.model_args.model_name_or_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = transformers.T5Tokenizer.from_pretrained("t5-base", return_dict=False)
        model_class = FiDT5
        self.model = model_class.from_pretrained(self.model_args.model_name_or_path)
        self.model = self.model.to(self.device)

        self.args = {
            "max_seq_length": 384,
            "doc_stride": 128,
            "max_query_length": 64,
            "threads": 1,
            "tqdm_enabled": False,
            "n_best_size": 20,
            "max_answer_length": 384,
            "do_lower_case": True,
            "output_prediction_file": False,
            "output_nbest_file": self.model_args.output_nbest_file,
            "output_null_log_odds_file": None,
            "verbose_logging": False,
            "version_2_with_negative": True,
            "null_score_diff_threshold": 0,
            "pad_on_right": False,
        }

    def update_args(self, args_to_change):
        for key in args_to_change:
            self.args[key] = args_to_change[key]

    def predict(self, question: Question, contexts: List[Context]) -> List[Answer]:

        eval_examples = [{
            "question": question.text,
            "ctxs": [{"title":x.title, "text": x.text, "score": x.score} for x in contexts],
            "id": 0,
            "target": "",
            "answers": [],
        }]

        eval_dataset = FidDataset(eval_examples, len(contexts))
        collator = Collator(self.args["max_seq_length"], self.tokenizer, answer_maxlength=self.args["max_answer_length"])

        sampler = SequentialSampler(eval_dataset)
        dataloader = DataLoader(eval_dataset,
                                sampler=sampler,
                                batch_size=self.model_args.eval_batch_size,
                                drop_last=False,
                                num_workers=10,
                                collate_fn=collator
                                )
        self.model.eval()

        answers = []
        if hasattr(self.model, "module"):
            self.model = self.model.module

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                (idx, _, _, context_ids, context_mask) = batch

                outputs = self.model.generate(
                    input_ids=context_ids.cuda(),
                    attention_mask=context_mask.cuda(),
                    max_length=50
                )

                for k, o in enumerate(outputs):
                    ans = self.tokenizer.decode(o, skip_special_tokens=True)
                    answers.append(Answer(
                        text=ans,
                        score=0,
                        # score=all_nbest_json[ans][0]["start_logit"] + all_nbest_json[ans][0]["end_logit"],
                        ctx_score=contexts[idx].score,
                        language=question.language
                    ))

        return answers

    #     all_answers = []
    #     for idx, ans in enumerate(all_nbest_json):
    #         all_answers.append(Answer(
    #             text=all_nbest_json[ans][0]["text"],
    #             score=all_nbest_json[ans][0]["probability"],
    #             # score=all_nbest_json[ans][0]["start_logit"] + all_nbest_json[ans][0]["end_logit"],
    #             ctx_score=contexts[idx].score,
    #             language=question.language
    #         ))
    #     return all_answers

