from finetune import Config, EmailDatasetPreprocessor, PrepareDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd

hypothesis_lst = list(Config.hypothesis_label_dic.values())
tokenizer = AutoTokenizer.from_pretrained(Config.FINETUNED_MODEL_PATH, model_max_length = Config.MAX_LEN)
model = AutoModelForSequenceClassification.from_pretrained(Config.FINETUNED_MODEL_PATH)
model.to(Config.device)

pipe_classifier = pipeline(
    "zero-shot-classification",
    model=model,  
    tokenizer=tokenizer,
    framework="pt",
    device=Config.device,
)

data = {"Subject" : ["[Ext] Splunk application update - Software Engineer Intern - Front-end (Boulder, CO - Summer 2024) (28270)"], "Body": ["Hi Vivekanand, Thank you so much for applying to the Software Engineer Intern - Front-end (Boulder, CO - Summer 2024) (28270 position at Splunk. We appreciate your interest and the time you’ve invested.Unfortunately, we have filled this position and will not move forward with your application at this time. If you applied to multiple positions at Splunk, note that you will receive a separate update for each one.We would encourage you to sign up for our job alerts and hope you’ll keep Splunk in mind for future opportunities.While we obviously can’t hire every applicant, we do strive to give every candidate a positive experience and we would love to hear about yours. Would you mind taking this short survey? It should only take a couple of minutes to complete. Your responses are confidential. Splunk will not know the identity of the respondents. Data will be used in aggregate to inform improvements to our processes.We wish you every success in your job search and in all of your professional pursuits.Sincerely,The Splunk Talent Acquisition Team Splunk’s Career Site Privacy Policy explains how we collect, use, store and share your information when you apply for a position at Splunk. This message is intended only for the personal, confidential, and authorized use of the recipient(s) named above. If you are not that person, you are not authorized to review, use, copy, forward, distribute or otherwise disclose the information contained in the message.."], "Label": ["Rejected"]}
df = pd.DataFrame(data)

preprocessor = EmailDatasetPreprocessor()
df["text"] = preprocessor.fit_hypithesis(df)
df = preprocessor.format_nli_testset(df,Config.hypothesis_label_dic)
text_lst = df["text"].tolist()

pipe_output = pipe_classifier(
    text_lst,  # input any list of texts here
    candidate_labels=hypothesis_lst,
    hypothesis_template="{}",
    multi_label=False,  # here you can decide if, for your task, only one hypothesis can be true, or multiple can be true
    batch_size=4  # reduce this number to 8 or 16 if you get an out-of-memory error
)
print(pipe_output)

hypothesis_pred_true_probability = []
hypothesis_pred_true = []
for dic in pipe_output:
    hypothesis_pred_true_probability.append(dic["scores"][0])
    hypothesis_pred_true.append(dic["labels"][0])

# map the long hypotheses to their corresponding short label names
hypothesis_label_dic_inference_inverted = {value: key for key, value in Config.hypothesis_label_dic.items()}
label_pred = [hypothesis_label_dic_inference_inverted[hypo] for hypo in hypothesis_pred_true]

df["label_text_pred"] = label_pred
df["label_text_pred_proba"] = hypothesis_pred_true_probability

print(df)