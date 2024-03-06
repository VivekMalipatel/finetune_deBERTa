from finetune import TriageDataset,Config, BERTClassifier
from transformers import BertTokenizer, logging
from torch.utils.data import DataLoader
from preprocess_data import EmailPreprocessor
import torch
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

logging.set_verbosity_error()

Config.BERT_PATH = 'bert_finetuned'

tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)

id2label = {0 :'Rejected', 1 :'Applied', 2 :'Irrelevant', 3: 'Accepted'}

data = {"Body": ["Hey Vivekanand Reddy, My name is Marchem Pfeiffer and I'm the student outreach & engagement coordinator for the AI Futures program."], "ENCODE_CAT": [1]}
df = pd.DataFrame(data)
preprocess = EmailPreprocessor()
df = preprocess.preprocess_dataframe(df)

training_set = TriageDataset(df, tokenizer, Config.MAX_LEN)

train_params = {'batch_size': Config.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)

model = BERTClassifier().to(Config.device)
model.eval()

for _ in range(0,10):
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(Config.device, dtype=torch.long)
        mask = data['mask'].to(Config.device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(Config.device, dtype=torch.long)
        outputs = model(ids, mask, token_type_ids)
        big_val, big_idx = torch.max(outputs.data, dim=1)
        print(id2label[big_idx[0].item()])