# Function for formatting of test data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from datetime import datetime
import time
from klass import get_classification

def get_sn():
    """ Get sn codes, names and notes from SSBs klass."""
    sn = get_classification(6)
    sn_df = sn.get_codes().data
    sn_df = sn_df.loc[sn_df.level == '5', ["code","name","notes"]].reset_index(drop = True)
    return sn_df
    
def reduce_test(dt, min_nr = 2, state = 42, test_size = 5000):
    dt2 = dt.loc[dt.nace_21_code != "00.000",:]
    
    counts = dt2["nace_21_code"].value_counts()
    valid_classes = counts[counts >= min_nr].index
    dt3 = dt2[dt2["nace_21_code"].isin(valid_classes)]
    
    _, test_small = train_test_split(
        dt3,
        test_size=test_size,
        stratify=dt3["nace_21_code"],
        random_state=42
    )
    test_small.reset_index(drop = True)
    return test_small
    

def write_results(classification_results, rag_type, max_nr, total_time, llm_model):
    n = len(classification_results)
    sn = get_sn()
    valid_codes = set(sn["code"])
    if rag_type in ("standard", "observed"):
        retrieved_hits = sum(
            r["observed_code"] in r["retrieved_codes"]
            for r in classification_results.values()
        )
        retrieved_pct = retrieved_hits / n

    else:
        retrieved_pct = ""
    valid_hits = sum(
        r["predicted"] in valid_codes
        for r in classification_results.values()
    )
    pred_in_hits = sum(
        r["predicted"] in r["retrieved_codes"]
        for r in classification_results.values()
    )
    
    y_true = [r["observed_code"] for r in classification_results.values()]
    y_pred = [r["predicted"] for r in classification_results.values()]
    y_true_4 = [y[:5] for y in y_true]
    y_pred_4 = [y[:5] for y in y_pred]
    
    exact4_pct = accuracy_score(y_true_4, y_pred_4)
    exact_pct = accuracy_score(y_true, y_pred)
    valid_pct = valid_hits / n
    pred_in_hits_pct = pred_in_hits / n
    
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

     # Write to file
    date_str = datetime.now().strftime("%Y-%m-%d_%H:%M")
    filename = f"results/results_rag_{rag_type}_{date_str}.txt"

    with open(filename, "w") as f:
        f.write(f"Classified {max_nr} of observations in the test data\n")
        f.write(f"Classified using model {llm_model['name']}\n")
        f.write(f"Classified using method {rag_type}\n")
        f.write(f"Retrieved percentage: {retrieved_pct}\n")
        f.write(f"Valid percentage: {valid_pct}\n")
        f.write(f"Predicted in retrieved percentage: {pred_in_hits_pct}\n")
        f.write(f"Accuracy: {exact_pct}\n")
        f.write(f"Accuracy (4.digits): {exact4_pct}\n")
        f.write(f"F1-macro: {f1_macro}\n")
        f.write(f"F1-weighted: {f1_weighted}\n")
        f.write(f"Run time: {round(total_time/60, 2)} minutes\n")

    print(f'File written to {filename}')
