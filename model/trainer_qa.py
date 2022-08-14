from transformers import BertTokenizer, BertForTokenClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import Dataset
from typing import Tuple, List, Optional, Dict

class SpacingTrainer(Trainer):
    def __init__(self, *args, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_process_function = post_process_function

    def predict(self, test_dataset: Dataset, test_examples, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ) -> Dict:

        prediction_output = {}
        test_dataloader = self.get_test_dataloader(test_dataset)

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, 
            description="Prediction", 
            ignore_keys=ignore_keys, 
            metric_key_prefix=metric_key_prefix
        )

        if self.post_process_function is None:
            return output._asdict()

        text_prediction = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args, self.args.output_dir
        )

        prediction_output = output._asdict()
        prediction_output['text_prediction'] = text_prediction

        return prediction_output

        
