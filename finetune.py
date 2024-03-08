import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig, __version__
import datasets
import torch
import warnings
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support, accuracy_score
warnings.filterwarnings('ignore')
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class Config:

    TRANSFORMER_VERSION = __version__

    BASE_MODEL_PATH = 'deberta-v3-xsmall-zeroshot-v1.1-all-33'
    DATA_FILE_PATH = 'preprocessed_emails_overSampled.csv'
    LOGS_PATH = 'DeBERTa_xsmall_finetuned_logs'
    FINETUNED_SAVE_PATH = "applicationTracker_DeBERTa_v3_xsmall_finetuned"
    MODEL_FILE_NAME = '/pytorch_model.bin'
    VOCAB_FILE_NAME = '/vocab.txt'
    OUTPUT_MODEL_NAME = FINETUNED_SAVE_PATH

    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 4 if "large" in BASE_MODEL_PATH else 8
    VALID_BATCH_SIZE = 64 if "large" in BASE_MODEL_PATH else 64*2
    EPOCHS = 3
    LEARNING_RATE = 9e-6 if "large" in BASE_MODEL_PATH else 2e-5
    WARMUP_RATIO = 0.06
    WEIGHT_DECAY = 0.01
    GRADIENT_ACCUMULATION_STEPS = 4 if "large" in BASE_MODEL_PATH else 1
    MODEL_TORCH_D_TYPE = "float16"
    FP16_BOOL = True if torch.cuda.is_available() else False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_size = 0.2
    SEED_GLOBAL = 42
    np.random.seed(SEED_GLOBAL)
    torch.manual_seed(SEED_GLOBAL)

    hypothesis_label_dic = {
    "Applied": "The email is related to a job application that the recipient has submitted, for instance, a confirmation email received after applying for a job.",
    "Rejected": "The email is related to a rejection from a job application, indicating that the recipient was not selected for the job role or that the application will not be moving forward.",
    "Irrelevant": "The email is not related to job applications, such as applying, being rejected, or being accepted for a job role. It does not pertain to the status or process of job applications.",
    "Accepted": "The email is related to the acceptance of a job application, indicating that the recipient has not just applied but been selected or accepted for a job role following an application."
    }

class EmailDatasetPreprocessor:
    def __init__(self, file_path = None):
        self.file_path = file_path
        #self.encode_dict = {'Rejected': 0, 'Applied': 1, 'Irrelevant': 2, 'Accepted' :3}
        
    def encode_cat(self, x):
        if x not in self.encode_dict.keys():
            self.encode_dict[x] = len(self.encode_dict)
        return self.encode_dict[x]
    
    def train_test_split(self, df):
        
        train_dataset = df.sample(frac=Config.train_size, random_state=Config.SEED_GLOBAL).reset_index(drop=True)
        test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

        return train_dataset,test_dataset
    
    def fit_hypothesis(self, email):
        return email.Subject.fillna("") + '. The email: "' + email.Body.fillna("") + '" -end of the email. '
    
    def format_nli_trainset(self, df, hypo_label_dic, random_seed=42):
        print(f"Length of Train df before formatting step: {len(df)}.")
        length_original_data = len(df)

        df_lst = []
        for label_text, hypothesis in hypo_label_dic.items():
            ## entailment
            df_step = df[df.Label == label_text].copy(deep=True)
            df_step["hypothesis"] = [hypothesis] * len(df_step)
            df_step["label"] = [0] * len(df_step)
            ## not_entailment
            df_step_not_entail = df[df.Label != label_text].copy(deep=True)
            df_step_not_entail = df_step_not_entail.sample(n=min(len(df_step), len(df_step_not_entail)), random_state=random_seed)
            df_step_not_entail["hypothesis"] = [hypothesis] * len(df_step_not_entail)
            df_step_not_entail["label"] = [1] * len(df_step_not_entail)
            # append
            df_lst.append(pd.concat([df_step, df_step_not_entail]))
        df = pd.concat(df_lst)

        # shuffle
        df = df.sample(frac=1, random_state=random_seed)
        df["label"] = df.label.apply(int)
        df["label_nli_explicit"] = ["True" if label == 0 else "Not-True" for label in df["label"]]  # adding this just to simplify readibility

        print(f"After adding not_entailment training examples, the training data was augmented to {len(df)} texts.")
        print(f"Max augmentation could be: len(df_train) * 2 = {length_original_data*2}. It can also be lower, if there are more entail examples than not-entail for a majority class.")

        return df.copy(deep=True)
    
    def format_nli_testset(self, df, hypo_label_dic):
        ## explode test dataset for N hypotheses
        hypothesis_lst = [value for key, value in hypo_label_dic.items()]
        print("Number of hypotheses/classes: ", len(hypothesis_lst))

        # label lists with 0 at alphabetical position of their true hypo, 1 for not-true hypos
        label_text_label_dic_explode = {}
        for key, value in hypo_label_dic.items():
            label_lst = [0 if value == hypo else 1 for hypo in hypothesis_lst]
            label_text_label_dic_explode[key] = label_lst

        df["label"] = df.Label.map(label_text_label_dic_explode)
        df["hypothesis"] = [hypothesis_lst] * len(df)
        print(f"Original test set size: {len(df)}")

        # explode dataset to have K-1 additional rows with not_entail label and K-1 other hypotheses
        # ! after exploding, cannot sample anymore, because distorts the order to true label values, which needs to be preserved for evaluation code
        df = df.explode(["hypothesis", "label"])  # multi-column explode requires pd.__version__ >= '1.3.0'
        print(f"Test set size for NLI classification: {len(df)}\n")

        df["label_nli_explicit"] = ["True" if label == 0 else "Not-True" for label in df["label"]]  # adding this just to simplify readibility

        return df.copy(deep=True)
    
    def read_input_file(self):
        df = pd.read_csv(self.file_path, encoding='utf-8')
        return df[['Subject','Body', 'Label']]
    
    def preprocess(self):
        df = self.read_input_file()
        #df['ENCODE_CAT'] = df['Label'].apply(lambda x: self.encode_cat(x))
        df["text"] = self.fit_hypothesis(df)
        train_df, test_df = self.train_test_split(df)
        train_df = self.format_nli_trainset(train_df, Config.hypothesis_label_dic, Config.SEED_GLOBAL)
        test_df = self.format_nli_testset(test_df, Config.hypothesis_label_dic)
        return train_df,test_df
    
class PrepareDataset:
    def __init__(self, train_df, test_df, tokenizer, max_len):
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def convert_to_dataset(self):
        columns_to_keep = ["text", "hypothesis", "label"]
        return datasets.DatasetDict({"train" : datasets.Dataset.from_pandas(self.train_df[columns_to_keep]),
                                    "test" : datasets.Dataset.from_pandas(self.test_df[columns_to_keep])})
    
    def tokenize_nli_format(self, examples):
        return self.tokenizer(examples["text"], examples["hypothesis"], padding=True, truncation = True, max_length = self.max_len, return_tensors="pt")
    
    def prepareDataset(self):
        dataset = self.convert_to_dataset()
        dataset = dataset.map(self.tokenize_nli_format, batched=True)
        return dataset
    
class Train:
    def __init__(self, model, tokenizer, dataset, train_df):

        self.model = model
        self.tokenizer = tokenizer
        self.dattaset = dataset
        self.train_df = train_df

        self.train_args = TrainingArguments(
            output_dir=Config.LOGS_PATH,
            logging_dir=Config.LOGS_PATH,
            learning_rate=Config.LEARNING_RATE,
            lr_scheduler_type= "linear",
            group_by_length=False, 
            gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
            per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=Config.VALID_BATCH_SIZE,
            num_train_epochs=Config.EPOCHS,
            warmup_ratio=Config.WARMUP_RATIO,
            weight_decay=Config.WEIGHT_DECAY,
            fp16=Config.FP16_BOOL, 
            seed = Config.SEED_GLOBAL,
            load_best_model_at_end= True,
            metric_for_best_model="f1_macro",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit = 2,
            report_to="all",
        )
        self.trainer = Trainer(
            model = self.model,
            tokenizer = self.tokenizer,
            args = self.train_args,
            train_dataset = self.dattaset["train"],
            eval_dataset = self.dattaset["test"],
            compute_metrics = lambda eval_pred : Eval(self.train_df).compute_metrics_nli_binary(eval_pred),
        )

    def Train(self):
        self.trainer.train()
        self.save_model()

    def save_model(self):
        model.config._name_or_path = Config.OUTPUT_MODEL_NAME
        self.trainer.model.config._name_or_path = Config.OUTPUT_MODEL_NAME
        self.trainer.model.config.transformers_version = Config.TRANSFORMER_VERSION
        self.trainer.model.config.torch_dtype = Config.MODEL_TORCH_D_TYPE
        self.trainer.save_model(output_dir=Config.FINETUNED_SAVE_PATH)
        torch.save(self.trainer.model, Config.FINETUNED_SAVE_PATH + Config.MODEL_FILE_NAME, _use_new_zipfile_serialization=False)
        self.trainer.tokenizer.save_vocabulary(Config.FINETUNED_SAVE_PATH)

class Eval:
    def __init__(self, df_train):
        self.df_train = df_train
        self.label_text_alphabetical = np.sort(self.df_train.Label.unique())
        

    def compute_metrics_nli_binary(self, eval_pred):

        predictions, labels = eval_pred

        ### reformat model output to enable calculation of standard metrics
        # split in chunks with predictions for each hypothesis for one unique premise
        def chunks(lst, n):  # Yield successive n-sized chunks from lst. https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        # for each chunk/premise, select the most likely hypothesis
        softmax = torch.nn.Softmax(dim=1)
        prediction_chunks_lst = list(chunks(predictions, len(set(self.label_text_alphabetical)) ))
        hypo_position_highest_prob = []
        for i, chunk in enumerate(prediction_chunks_lst):
            hypo_position_highest_prob.append(np.argmax(np.array(chunk)[:, 0]))  # only accesses the first column of the array, i.e. the entailment/true prediction logit of all hypos and takes the highest one

        label_chunks_lst = list(chunks(labels, len(set(self.label_text_alphabetical)) ))
        label_position_gold = []
        for chunk in label_chunks_lst:
            label_position_gold.append(np.argmin(chunk))  # argmin to detect the position of the 0 among the 1s

        #print("Highest probability prediction per premise: ", hypo_position_highest_prob)
        #print("Correct label per premise: ", label_position_gold)

        ### calculate standard metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='macro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(label_position_gold, hypo_position_highest_prob, average='micro')  # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        acc_balanced = balanced_accuracy_score(label_position_gold, hypo_position_highest_prob)
        acc_not_balanced = accuracy_score(label_position_gold, hypo_position_highest_prob)
        metrics = {
            'accuracy': acc_not_balanced,
            'f1_macro': f1_macro,
            'accuracy_balanced': acc_balanced,
            'f1_micro': f1_micro,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            #'label_gold_raw': label_position_gold,
            #'label_predicted_raw': hypo_position_highest_prob
        }
        #print("Aggregate metrics: ", {key: metrics[key] for key in metrics if key not in ["label_gold_raw", "label_predicted_raw"]} )  # print metrics but without label lists
        #print("Detailed metrics: ", classification_report(label_position_gold, hypo_position_highest_prob, labels=np.sort(pd.factorize(label_text_alphabetical, sort=True)[0]), target_names=label_text_alphabetical, sample_weight=None, digits=2, output_dict=True,
        #                            zero_division='warn'), "\n")
        return metrics

if __name__ == "__main__":

    preprocessor = EmailDatasetPreprocessor(Config.DATA_FILE_PATH)
    train_df, test_df = preprocessor.preprocess()

    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_PATH, model_max_length = Config.MAX_LEN)

    model_config = AutoConfig.from_pretrained(Config.BASE_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(Config.BASE_MODEL_PATH, config = model_config)
    model.to(Config.device)
    model.config.transformers_version = Config.TRANSFORMER_VERSION
    Config.MODEL_TORCH_D_TYPE = model_config.torch_dtype

    dataset = PrepareDataset(train_df, test_df, tokenizer, Config.MAX_LEN).prepareDataset()

    print("The overall structure of the pre-processed train and test sets:\n")
    print(dataset)

    trainer = Train(model,tokenizer,dataset,train_df)
    trainer.Train()

