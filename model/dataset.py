import torch
import pandas as pd
from typing import List, Tuple, Optional
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, transform, data_path:Optional[str]=None, sentence:Optional[str]=None):
        self.data_path = data_path
        self.slot_labels = ["UNK", "PAD", "B", "I"]
        self.transform = transform

        if sentence:
            self.sentences = [sentence.split()]
            self.ground_truths = self.sentences
        if data_path:
            self.load_data()

    def load_data(self):
        df = pd.read_csv(self.data_path, sep=',')
        sentences = df['wrong_sentence'].to_list()
        ground_truths = df['correct_sentence'].to_list()

        self.sentences = [line.split() for line in sentences]
        self.ground_truths = [line.split() for line in ground_truths]

    def get_tags(self, sentence: List[str]) -> List[str]:
        all_tags = []
        for word in sentence:
            for i, c in enumerate(word):
                if i == 0:
                    all_tags.append("B")
                else:
                    all_tags.append("I")

        return all_tags

    def getlabels(self):
        gt_list = []
        for gt in self.ground_truths:
            tags = self.get_tags(gt)
            tags = [self.slot_labels.index(t) for t in tags]
            gt_list.append(tags)
        return gt_list

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = "".join(self.sentences[idx])
        tags = self.get_tags(self.sentences[idx])
        tags = [self.slot_labels.index(t) for t in tags]

        target_tags = self.get_tags(self.ground_truths[idx])
        target_tags = [self.slot_labels.index(t) for t in target_tags]

        (
            input_ids,
            attention_mask,
            token_type_ids,
            targets
        ) = self.transform(sentence, tags, target_tags)

        return {
          'input_ids': input_ids,
          'attention_mask': attention_mask,
          'token_type_ids': token_type_ids, # distilbert 사용 할 때 주석 처리
          'labels': targets
        }