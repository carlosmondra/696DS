import pickle
import torch
import json
from torch.utils.data import Dataset, DataLoader
from bert_serving.client import BertClient

import time
import numpy as np

class PaperDataset(Dataset):
    """Any Scientific Papers dataset."""

    def __init__(self, papers_path, idx_to_bytes):
        """
        Args:
            papers_path (string): Path to the file with papers in JSON format.
            idx_to_bytes (string): Array that maps index to byte position
            in the file with the papers in JSON format
        """
        self.papers_path = papers_path
        self.idx_to_bytes = idx_to_bytes
        self.bc = BertClient(check_length=False)

    def __len__(self):
        return len(self.idx_to_bytes)

    def __getitem__(self, idx):
        with open(self.papers_path) as papers:
            bytes_pos = self.idx_to_bytes[idx]
            papers.seek(bytes_pos)
            paper = json.loads([next(papers) for x in range(1)][0])
            s = time.time()
            embed_sections = []
            for section in paper['sections']:
                sent_embs = self.bc.encode(section)
                embed_sections.append(sent_embs) 
        # These are the keys of each paper:
        # article_id
        # article_text
        # abstract_text
        # labels -> has None
        # section_names
        # sections -> Same as article_text, but divided in sections

        return {'sections': embed_sections, 'type': type(sent_embs), 'time': time.time() - s}

idx_to_bytes = pickle.load(open('arxiv-release/val_idx_to_bytes.pkl', 'rb'))
paper_dataset = PaperDataset('arxiv-release/val.txt', idx_to_bytes)
# Show paper in line 1001:
print(paper_dataset[1000]['time'])
print(paper_dataset[1000]['sections'][0][0])
