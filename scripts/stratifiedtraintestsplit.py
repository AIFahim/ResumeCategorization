import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import os
from textcleaner import TextCleaner


class StratifiedTrainValTestSplit:
    def __init__(self, data_dir='data', test_size=0.2, val_size=0.1, random_state=42):
        self.data_dir = data_dir
        self.resume_csv_path = os.path.join(data_dir, 'Resume.csv')
        self.train_data_path = os.path.join(data_dir, 'train_data.csv')
        self.val_data_path = os.path.join(data_dir, 'val_data.csv')
        self.test_data_path = os.path.join(data_dir, 'test_data.csv')
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.text_cleaner = TextCleaner()  # Initialize TextCleaner

    def load_data(self):
        # Load the dataset
        self.resume_data = pd.read_csv(self.resume_csv_path)

    def preprocess_data(self):
        # Clean and preprocess the data using TextCleaner
        self.resume_data['cleaned_resume'] = self.resume_data['Resume_str'].apply(lambda x: self.text_cleaner.clean_text(x))

    def stratified_split(self):
        # Split the dataset into train and temp (temp will be split into validation and test)
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size + self.val_size, random_state=self.random_state)
        for train_index, temp_index in strat_split.split(self.resume_data, self.resume_data['Category']):
            self.train_set = self.resume_data.loc[train_index].reset_index(drop=True)
            self.temp_set = self.resume_data.loc[temp_index].reset_index(drop=True)

        # Split temp into validation and test
        strat_split_temp = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size / (self.test_size + self.val_size), random_state=self.random_state)
        for val_index, test_index in strat_split_temp.split(self.temp_set, self.temp_set['Category']):
            self.val_set = self.temp_set.loc[val_index].reset_index(drop=True)
            self.test_set = self.temp_set.loc[test_index].reset_index(drop=True)

    def save_data(self):
        # Save the processed training, validation, and test sets
        self.train_set.to_csv(self.train_data_path, index=False)
        self.val_set.to_csv(self.val_data_path, index=False)
        self.test_set.to_csv(self.test_data_path, index=False)

    def run(self):
        print("Loading data...")
        self.load_data()
        print("Preprocessing data...")
        self.preprocess_data()
        print("Performing stratified train-validation-test split...")
        self.stratified_split()
        print("Saving processed data...")
        self.save_data()
        print("Data split complete. Training, validation, and test data saved.")

# Usage
if __name__ == "__main__":
    stratified_splitter = StratifiedTrainValTestSplit()
    stratified_splitter.run()
