import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import string

class Config:
    input_csv_path = "labeled_emails.csv"
    target_csv_path = "preprocessed_emails_underSampled.csv"

class EmailPreprocessor:
    def __init__(self):

        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        # Method to clean and preprocess a single piece of text
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\S*@\S*\s?', '', text)
        text = re.sub(f"[{string.punctuation}]", "", text)
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)
    
    def preprocess_dataframe(self, df, subject_col='Subject', body_col='Body'):
        if subject_col in df:
            df[subject_col] = df[subject_col].apply(lambda x: self.preprocess_text(x) if isinstance(x, str) else x)
        if body_col in df:
            df[body_col] = df[body_col].apply(lambda x: self.preprocess_text(x) if isinstance(x, str) else x)
        return df
    
    def underSample(self, df, label_col='Label'):
        # Method to balance the dataset
        ros = RandomUnderSampler(random_state=42)
        X = df.drop(label_col, axis=1)  # Features
        y = df[label_col]  # Target
        X_resampled, y_resampled = ros.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[label_col] = y_resampled
        return df_resampled

    def overSample(self, df, label_col='Label'):
        # Method to balance the dataset
        ros = RandomOverSampler(random_state=42)
        X = df.drop(label_col, axis=1)  # Features
        y = df[label_col]  # Target
        X_resampled, y_resampled = ros.fit_resample(X, y)
        df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        df_resampled[label_col] = y_resampled
        return df_resampled
    
    def read_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        return df
    
    def save_csv(self, df, output_csv_path):
        df.to_csv(output_csv_path, index=False, encoding='utf-8')

if __name__ == "__main__":
    # Ensure necessary NLTK data is downloaded
    #download('punkt')
    #download('stopwords')
    #download('wordnet')

    preprocessor = EmailPreprocessor()
    df = preprocessor.read_csv(Config.input_csv_path)
    df = preprocessor.preprocess_dataframe(df, subject_col='Subject', body_col='Body')
    df_balanced = preprocessor.underSample(df, label_col='Label')
    preprocessor.save_csv(df_balanced, Config.target_csv_path)