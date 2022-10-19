
import pickle
import pandas as pd
import numpy as np

test_data   = pd.read_pickle("../../all_lead1_test_processed.pickle")
test_label  = pd.read_csv("../../test_labels.csv")
test_data   = test_data.reset_index(drop=True)

print(test_data)

pace_mask   = test_label["category"] == "5"
idx         = np.array(range(len(pace_mask)))
idx_pace    = idx[np.array(pace_mask)]
idx_keep    = np.random.choice(idx_pace, size=5, replace=False)
idx_discard = np.setdiff1d(idx_pace, idx_keep)
test_data   = test_data.drop(index=idx_discard)
test_label  = test_label.drop(index=idx_discard)


print(test_data)
print(test_label)

test_data.to_pickle("all_lead1_test_processed.pickle")
test_label.to_csv("test_labels.csv")

