from finetune import Config, EmailDatasetPreprocessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from sklearn.metrics import accuracy_score

hypothesis_lst = list(Config.hypothesis_label_dic.values())
MODEL_PATH = "DeBERTa_large_finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, model_max_length = Config.MAX_LEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

pipe_classifier = pipeline(
    "zero-shot-classification",
    model=model,  
    tokenizer=tokenizer,
    framework="pt",
    device=Config.device,
)

#data = {"Subject" : ["Your Seismic Application: Software Engineer Intern - Summer 2024"], "Body": ["Hi Vivekanand Reddy, We appreciate your interest in joining the Seismic team. We know how time consuming it can be searching for a new opportunity, so we really mean it when we say thanks for thinking of us.After reviewing your application, we’ve decided to move forward with other candidates for the Software Engineer Intern - Summer 2024 role."], "Label": ["Rejected"]}
#df = pd.DataFrame(data)

preprocessor = EmailDatasetPreprocessor("preprocessed_emails.csv")

df = preprocessor.read_input_file()
df["text"] = preprocessor.fit_hypithesis(df)
text_lst = df["text"].tolist()

pipe_output = pipe_classifier(
    text_lst,
    candidate_labels=hypothesis_lst,
    hypothesis_template="{}",
    multi_label=False,
    batch_size=64
)

hypothesis_pred_true_probability = []
hypothesis_pred_true = []
for dic in pipe_output:
    hypothesis_pred_true_probability.append(dic["scores"][0])
    hypothesis_pred_true.append(dic["labels"][0])

# map the long hypotheses to their corresponding short label names
hypothesis_label_dic_inference_inverted = {value: key for key, value in Config.hypothesis_label_dic.items()}
label_pred = [hypothesis_label_dic_inference_inverted[hypo] for hypo in hypothesis_pred_true]

df["label_pred"] = label_pred
df["label_pred_proba"] = hypothesis_pred_true_probability

print(accuracy_score(df['Label'],df["label_pred"]))