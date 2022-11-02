
import matplotlib
import matplotlib.pyplot as plt
import seaborn           as sns
import pickle
import numpy             as np
import pandas            as pd
import tqdm
import sklearn.metrics 
import ast
import sklearn

from sklearn.decomposition import PCA
from joypy                 import joyplot

def classify_signal(diagnosis, code_table):
    codes          = [code_table[diag_str] for diag_str in diagnosis]
    arrhythm_codes = list(filter(lambda code : ~np.isnan(code), codes))

    memberships    = {"cat{}".format(i): False for i in range(10)}
    if len(np.intersect1d(arrhythm_codes, [1, 3, 4, 5, 6, 7, 8, 9])) == 0:
        memberships["is_arrhythm"] = False
    else:
        memberships["is_arrhythm"] = True
    for code in arrhythm_codes:
        memberships["cat{}".format(int(code))] = True
    return memberships

def classify_results(it, data, code_table):
    data = data.to_dict("records")
    processed_rows = []
    for d in tqdm.tqdm(data, total=len(data)):
        row_entry   = {}
        mll         = np.float64(d["mll"])
        memberships = classify_signal(d["cadio"], code_table)
        for (k, v) in memberships.items():
            if k != "is_arrhythm":
                if v:
                    memberships[k] = mll
                else:
                    memberships[k] = np.NaN

        row_entry.update(memberships)
        row_entry["iter"]   = it
        row_entry["fukuda"] = d["fukuda"]

        if memberships["is_arrhythm"]:
            row_entry["arrhythm_mll"] = mll
        else:
            row_entry["normal_mll"] = mll
        row_entry["mll"] = mll
        #row_entry["mu"]  = np.float64(d["mu"][0])
        processed_rows.append(row_entry)
    return processed_rows
    
def ridgelineplot(df):
    labels  = [y if y%20==0 else None for y in np.unique(np.array(df["iter"]))]
    df      = df.rename(columns={"normal_mll":"Normal sinus rhythms", "arrhythm_mll":"Abnormal rhythms"})
    ax, fig = joyplot(
        data        = df,
        by          = 'iter',
        overlap     = 4,
        labels      = labels,
        column      = ["Normal sinus rhythms",
                       "Abnormal rhythms"],
                       #'cat0',
                       #'cat1',
                       #'cat2',
                       #'cat3',
                       #'cat4',
                       #'cat5',
                       #'cat6',
                       #'cat7',
                       #'cat8',
                       #'cat9',
                       #'cat10'],
        color=['#686de0', '#eb4d4b'],
        legend      = True,
        figsize     = (5, 7),
        linewidth   = 1,
        kind        = "kde", 
        alpha       = 0.4,
        x_range = [-5000,100]
    )
    plt.xlabel("Marginal log likelihood")
    plt.savefig("ridgeline.svg")

def histograms(df, categories, fukuda_table):
    df_last = df[df["iter"] == np.max(df["iter"])]
    f, axes = plt.subplots(len(categories), 1)
    print(len(df_last[~df_last["is_arrhythm"]]))
    for idx, id in enumerate(categories):
        df_subset   = df_last[~np.isnan(df_last["cat{}".format(id)]) | ~df_last["is_arrhythm"]]
        df_arrhythm = df_subset[df_subset["is_arrhythm"]]
        df_normal   = df_subset[~df_subset["is_arrhythm"]]

        roc_auc = sklearn.metrics.roc_auc_score(df_subset["is_arrhythm"], -df_subset["mll"])
        print("cat {} --------------".format(id))
        print("count    = {}".format(len(df_arrhythm)))
        print("ROC AUC  = {:.4f}".format(roc_auc))

        ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(df_subset["is_arrhythm"], -df_subset["mll"])
        thres = 0.95
        tpr_thres = np.argmax(1 - ns_fpr < thres) 
        fpr_thres = np.argmax(ns_tpr > thres) 
        print("sensitivity = {:.4f}".format(ns_tpr[tpr_thres]))
        print("specificity = {:.4f}".format(1 - ns_fpr[fpr_thres]))

        pr, rc, _ = sklearn.metrics.precision_recall_curve(df_subset["is_arrhythm"], -df_subset["mll"])
        pr_auc    = sklearn.metrics.auc(rc, pr)
        print("PRC AUC  = {:.4f}".format(pr_auc))

        f1s = 2*rc*pr/(rc+pr + 1e-10)
        print("F1 score = {:.4f}".format(np.max(f1s)))

        true_normal = df_subset["is_arrhythm"] 
        pred_normal = ~np.array([(code in fukuda_table.keys()) and (fukuda_table[code] == 0) for code in df_subset["fukuda"]])
        tp = np.sum(true_normal*pred_normal)
        fp = np.sum((~true_normal)*pred_normal)
        tn = np.sum((~true_normal)*(~pred_normal))
        fn = np.sum((true_normal)*(~pred_normal))

        f1_pos = 2*tp / (2*tp +  fp + fn + 1e-10)
        print("fukuda sensitivity = {:.4f}".format(tp/(tp + fn)))
        print("fukuda specificity = {:.4f}".format(tn/(tn + fp)))
        print("fukuda F1 positive = {:.4f}".format(f1_pos))
        print("fukuda recall      = {:.4f}".format(sklearn.metrics.recall_score(   true_normal, pred_normal)))
        print("fukuda precision   = {:.4f}".format(sklearn.metrics.precision_score(true_normal, pred_normal)))

        sns.kdeplot(df_arrhythm["mll"], shade=True, label="cat{}".format(id), ax=axes[idx])
        sns.kdeplot(df_normal["mll"],   shade=True, label="normal",           ax=axes[idx])
        axes[idx].legend()
    print("------------------------------")
    plt.legend(frameon=False)
    plt.show()
    
def evaluate_performance(df, fukuda_table, prefix):
    df      = df[df["iter"] == np.max(df["iter"])]

    true_normal = df["is_arrhythm"] 
    pred_normal = ~np.array([(code in fukuda_table.keys()) and (fukuda_table[code] == 0) for code in df["fukuda"]])
    tp = np.sum(true_normal*pred_normal)
    fp = np.sum((~true_normal)*pred_normal)
    tn = np.sum((~true_normal)*(~pred_normal))
    fn = np.sum((true_normal)*(~pred_normal))

    f1_pos = 2*tp / (2*tp +  fp + fn)
    f1_neg = 2*tn / (2*tn +  fn + fp)
    fukuda_sensitivity = tp/(tp + fn)
    fukuda_specificity = tn/(tn + fp)
    print("fukuda sensitivity = {:.4f}".format(fukuda_sensitivity))
    print("fukuda specificity = {:.4f}".format(fukuda_specificity))
    print("fukuda F1 positive = {:.4f}".format(f1_pos))
    print("fukuda precision   = {:.4f}".format(sklearn.metrics.precision_score(true_normal, pred_normal)))
    print("fukuda recall      = {:.4f}".format(sklearn.metrics.recall_score(   true_normal, pred_normal)))

    y_label = 1 - np.array(df["is_arrhythm"]).astype(np.int)
    score   = np.array(df["mll"]).astype(np.float64)

    np.save("score_{}.npy".format(prefix), score)
    np.save("labels_{}.npy".format(prefix), y_label)

    roc_auc = sklearn.metrics.roc_auc_score(y_label, score)
    print("ROC - area under curve = ", roc_auc)

    pr, rc, _ = sklearn.metrics.precision_recall_curve(y_label, score)
    pr_auc    = sklearn.metrics.auc(rc, pr)
    
    print("PR  - area under curve = ", pr_auc)

    f1s = 2*rc*pr/(rc+pr + 1e-10)
    print("F1 score = ", np.max(f1s))
    #print("recall   = {:.4f}".format(sklearn.metrics.recall_score(true_normal, pred_normal)))

    plt.plot(rc, pr, label="PR curve")
    plt.xlabel("recall (sensitivity)")
    plt.ylabel("precision")
    plt.legend()
    plt.show()

    ns_fpr, ns_tpr, _ = sklearn.metrics.roc_curve(y_label, score)

    fpr_thres = 0.90
    tpr_thres = 0.90
    tpr_thres = np.argmax(1 - ns_fpr < fpr_thres) 
    fpr_thres = np.argmax(ns_tpr > tpr_thres) 
    print("sensitivity = {:.4f}".format(ns_tpr[tpr_thres]))
    print("specificity = {:.4f}".format(1 - ns_fpr[fpr_thres]))

    plt.plot(ns_fpr, ns_tpr, label="ROC curve")
    plt.xlabel("false positive rate")
    plt.ylabel("true  positive rate")
    plt.legend()
    plt.show()

    return

def fetch_process_all(iter_range):
    idx = 0
    df  = pd.DataFrame(columns=['iter', 'arrhythm_mll', 'normal_mll', 'mll', 'is_arrhythm'])
    processed_rows = []
    for i in tqdm.tqdm(iter_range, total=len(iter_range)):
        with open("src/samples/{}/validation.pickle".format(i), "rb") as f:
            processed_rows.extend(
                classify_results(i, pickle.load(f)))
    return pd.DataFrame(processed_rows)

def samsung_diagnostic_table():
    classes    = pd.read_csv("../diagnostic_counts_final_category.csv")
    codes      = classes[["diagnosis", "arrhythmia_code"]].to_dict("list")
    code_table = {codes["diagnosis"][i] : codes["arrhythmia_code"][i] for i in range(len(classes))}
    return code_table

def fukuda_diagnostic_table():
    classes    = pd.read_excel("../category.xlsx")
    codes      = classes[["code", "arrhythmia"]].to_dict("list")
    code_table = {str(codes["code"][i]) : codes["arrhythmia"][i] for i in range(len(classes))}
    return code_table

def main():
    matplotlib.rc('font', family='sans-serif') 
    matplotlib.rc('font', serif='Helvetica') 
    matplotlib.rc('text', usetex='false') 
    matplotlib.rcParams.update({'font.size': 9})

    samsung_table = samsung_diagnostic_table()
    fukuda_table  = fukuda_diagnostic_table()

    data = pd.read_pickle("src/all_lead1_test.gzip")
    #data = pd.read_pickle("src/all_lead1_gold_test.pickle")
    data = pd.DataFrame(classify_results(200, data, samsung_table))

    print(data[["cat{}".format(i) for i in range(0,11)]].count())

    subset = data[["cat{}".format(i) for i in [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]] + ["mll"]]
    subset.to_csv("categories_test.csv")

    #df = fetch_process_all([200]) #np.arange(0, 201, 10))
    #ridgelineplot(df)

    histograms(data, [0, 1, 2, 3, 4, 6], fukuda_table)
    #histograms(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 10])
    evaluate_performance(data, fukuda_table, 'test')
    return

if __name__ == "__main__":
    main()
