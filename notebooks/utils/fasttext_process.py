# Functions for running fasttext baseline model for comparisons and for including in augmented prompt
import numpy as np
import pandas as pd
import json
import os
import string
import re
from bs4 import BeautifulSoup
from io import StringIO
import requests

from sklearn.metrics import f1_score, accuracy_score

from faker import Faker
from faker.providers.person.no_NO import Provider as NoProvider


def get_names():
    """ Get list of common Norwegian names."""
    fake = Faker("no_NO")

    first_names = NoProvider.first_names
    last_names = NoProvider.last_names
    all_names = first_names + last_names
    
    all_lower = [s.lower() for s in all_names]

    return all_lower

def general_preprocess(series, names):
   
    # Convert to string and lowercase
    s = series.fillna("").astype(str).str.lower()

    # Remove HTML tags
    s = s.apply(lambda x: BeautifulSoup(x, "html.parser").get_text())

    # Remove numbers and special characters (keep æøå)
    s = s.str.replace(r"\d+", "", regex=True)
    s = s.str.replace(r"[^\w\sæøå]", " ", regex=True)

    # Remove names (vectorized)
    pattern_names = r'\b(' + '|'.join(names) + r')\b'
    s = s.str.replace(pattern_names, '', regex=True)

    # Remove any 'nan'
    s = s.str.replace(r'\b[nN][aA][nN]?\b', '', regex=True)
    
    # Remove extra spaces
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()

    return s

def cleaning_df(df:pd.DataFrame) -> pd.DataFrame:

    all_names = get_names()
    
    df.copy()

    df['company_activity'] = general_preprocess(df['company_activity'].fillna(' '), all_names)
    df['company_name'] = general_preprocess(df['company_name'].fillna(' '), all_names)
    df['company_purpose'] = general_preprocess(df['company_purpose'].fillna(' '), all_names)
        
    return df

def fasttext_evaluate(test, labels_1, probs_1, labels_5, probs_5):
    """ Function for evaluating on test data.
    """
    # ----------- Top 1 prediction ----------------
    # Clean labels
    pred_labels = [label[0].replace('__label__', '') for label in labels_1] 
    true_labels = test['nace_21_code'].astype(str).tolist()
    y_true_4 = [y[:5] for y in true_labels]
    y_pred_4 = [y[:5] for y in pred_labels]

    # F1 scores for top 1
    f1_macro = f1_score(true_labels, pred_labels, average='macro')
    f1_weighted = f1_score(true_labels, pred_labels, average='weighted')

    # Accuracy
    exact_pct = accuracy_score(true_labels, pred_labels)
    exact4_pct = accuracy_score(y_true_4, y_pred_4)

    # Check validity
    sn = get_sn()
    valid_codes = set(sn["code"])
    valid_hits = sum(
            r in valid_codes
            for r in pred_labels
        )
    valid_pct = valid_hits / len(true_labels)

    # ---------- Top 5 prediction -------------
    # Clean top-5 labels
    pred_labels_5 = [
        [label.replace('__label__', '') for label in row_labels]
        for row_labels in labels_5
    ]
    
    # Check if the correct label is in the top 5
    top5_correct = [
        true in preds
        for true, preds in zip(true_labels, pred_labels_5)
    ]
    top5_accuracy = sum(top5_correct) / len(top5_correct)

    # total time
    total_time = (time.time() - t_start)/60

    return {
        'valid_pct': valid_pct,
        'exact_pct': exact_pct,
        'exact4_pct': exact4_pct,
        'f1_macro': f1_macro,
        'f1_weighted':f1_weighted, 
        'top5_accuracy':top5_accuracy,
        'total_time': total_time,
    }