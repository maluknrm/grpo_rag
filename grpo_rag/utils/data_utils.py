"""Utils for data management"""
import pandas as pd
import numpy as np
from grpo_rag.utils.constants import DATA_PATH, DATA_X_COLUMN, DATA_Y_COLUMN
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, sample = int | None, random_state: int | None = None):
        self.df = pd.read_csv(DATA_PATH)
        if sample:
            self.df = self.sample_df(sample)
        self.random_state = random_state


    def sample_df(self, sample):
        return self.df[:sample]


    def get_train_test_set(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.df[DATA_X_COLUMN], self.df[DATA_Y_COLUMN],
            test_size=0.33, 
            random_state=self.random_state
        )
        
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)