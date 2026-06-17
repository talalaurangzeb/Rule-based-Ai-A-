"""Credit Risk Analysis Project
Decision Tree and Rule-Based Classification
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from collections import Counter
import os


RANDOM_STATE = 42
DATA_ROWS = 2000
OUTPUT_DIR = "./output_credit_rules"
os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_STATE)


def generate_credit_data(n=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    age = rng.randint(18, 75, size=n)
    annual_income = rng.normal(40000, 20000, size=n).clip(5000, 200000)
    credit_amount = rng.normal(15000, 10000, size=n).clip(500, 100000)
    duration = rng.randint(6, 84, size=n)
    employment_years = rng.randint(0, 40, size=n)
    previous_defaults = rng.choice([0,1,2,3], size=n, p=[0.85,0.1,0.03,0.02])
    num_dependents = rng.randint(0, 6, size=n)
    purpose = rng.choice(['car','education','furniture','business','house','other'],
                         size=n, p=[0.2,0.15,0.15,0.2,0.2,0.1])

    debt_to_income = credit_amount / (annual_income + 1.0)

    # Hidden generative risk formula (probabilistic)
    risk_score = (
        0.30*(previous_defaults) +
        0.25*(debt_to_income*10) +
        0.15*(credit_amount/40000) +
        0.15*(duration/60) -
        0.10*(employment_years/10) -
        0.05*(annual_income/100000)
    )

    risk_prob = 1.0 / (1.0 + np.exp(-(risk_score - 0.5 + rng.normal(0, 0.25, n))))
    target = (risk_prob > 0.5).astype(int)  # 1 = high risk, 0 = low risk

    df = pd.DataFrame({
        'age': age,
        'annual_income': np.round(annual_income, 2),
        'credit_amount': np.round(credit_amount, 2),
        'duration_months': duration,
        'employment_years': employment_years,
        'previous_defaults': previous_defaults,
        'num_dependents': num_dependents,
        'purpose': purpose,
        'debt_to_income': np.round(debt_to_income, 4),
        'high_risk': target
    })
    return df


df = generate_credit_data(DATA_ROWS, random_state=RANDOM_STATE)
# Quick basic sanity prints
print("Dataset shape:", df.shape)
print("Target distribution:\n", df['high_risk'].value_counts(normalize=False))
print(df.head())

# Split features/target
X = df.drop(columns=['high_risk'])
y = df['high_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------
# 3. Expert Rule-based Classifier (hand-crafted)
# ---------------------------
def expert_rule_predict_one(row):
    """
    Evaluate expert rules on a single applicant (Pandas Series).
    Returns tuple (prediction, rule_description)
    Ordered rules: first matching rule decides the classification.
    """
    # Rule 1: multiple previous defaults -> high risk
    if row['previous_defaults'] >= 2:
        return 1, "previous_defaults >= 2"
    # Rule 2: one default plus high debt-to-income ratio
    if row['previous_defaults'] == 1 and row['debt_to_income'] > 0.30:
        return 1, "previous_defaults == 1 and debt_to_income > 0.30"
    # Rule 3: very high debt-to-income
    if row['debt_to_income'] > 0.60:
        return 1, "debt_to_income > 0.60"
    # Rule 4: large loan with short employment history
    if row['credit_amount'] > 60000 and row['employment_years'] < 2:
        return 1, "credit_amount > 60000 and employment_years < 2"
    # Rule 5: long duration and low income
    if row['duration_months'] > 70 and row['annual_income'] < 30000:
        return 1, "duration_months > 70 and annual_income < 30000"
    # Rule 6: very low income but large credit
    if row['annual_income'] < 12000 and row['credit_amount'] > 15000:
        return 1, "annual_income < 12000 and credit_amount > 15000"
    # Default: low risk
    return 0, "default_low_risk"

def expert_rule_predict_df(X_df):
    preds = []
    rules = []
    for _, row in X_df.iterrows():
        p, r = expert_rule_predict_one(row)
        preds.append(int(p))
        rules.append(r)
    return np.array(preds), rules

expert_preds_test, expert_rules_used_test = expert_rule_predict_df(X_test)

# ---------------------------
# 4. Induced-rule classifier via Decision Tree
# ---------------------------
# One-hot encode categorical 'purpose' for tree-based model
X_train_tree = pd.get_dummies(X_train, columns=['purpose'], drop_first=True)
X_test_tree = pd.get_dummies(X_test, columns=['purpose'], drop_first=True)

# Ensure same columns between train and test
missing_cols = set(X_train_tree.columns) - set(X_test_tree.columns)
for c in missing_cols:
    X_test_tree[c] = 0
# Reorder test to match train columns
X_test_tree = X_test_tree[X_train_tree.columns]

# Fit a decision tree (interpretable rule inducer)
tree_clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=RANDOM_STATE)
tree_clf.fit(X_train_tree, y_train)

# Predictions & probabilities
tree_preds_test = tree_clf.predict(X_test_tree)
tree_probs_test = tree_clf.predict_proba(X_test_tree)[:, 1]

# Extract human-readable rules from the tree
tree_rules_text = export_text(tree_clf, feature_names=list(X_train_tree.columns), decimals=3)

# ---------------------------
# 5. Evaluation utilities & metrics
# ---------------------------
def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

expert_eval = evaluate(y_test, expert_preds_test)
tree_eval = evaluate(y_test, tree_preds_test)

# ROC & AUC
# For expert rules we approximate by giving probability 1.0 for predicted class and 0 for other class.
expert_probs_approx = expert_preds_test.astype(float)
fpr_tree, tpr_tree, _ = roc_curve(y_test, tree_probs_test)
auc_tree = auc(fpr_tree, tpr_tree)
fpr_exp, tpr_exp, _ = roc_curve(y_test, expert_probs_approx)
auc_exp = auc(fpr_exp, tpr_exp)

# ---------------------------
# 6. Interpretability / rule stats
# ---------------------------
expert_rule_set = [
    "previous_defaults >= 2",
    "previous_defaults == 1 & debt_to_income > 0.30",
    "debt_to_income > 0.60",
    "credit_amount > 60000 & employment_years < 2",
    "duration_months > 70 & annual_income < 30000",
    "annual_income < 12000 & credit_amount > 15000",
    "default_low_risk"
]
expert_num_rules = len(expert_rule_set)
expert_avg_tokens = np.mean([len(r.split()) for r in expert_rule_set])

tree_num_leaf_rules = tree_rules_text.count("class:")
# approximate clause complexity: average number of conditions per leaf derived from '|' depth
leaf_lines = [line for line in tree_rules_text.splitlines() if "class:" in line]
clauses_counts = []
for line in leaf_lines:
    # count '|' occurrences as depth indicators; treat each '|' presence as a clause
    clauses_counts.append(max(1, line.count("|")))
tree_avg_clause = float(np.mean(clauses_counts)) if clauses_counts else 0.0

# ---------------------------
# 7. Print summaries
# ---------------------------
print("\n--- Expert Rule System Evaluation (test set) ---")
for k, v in expert_eval.items():
    if k != "confusion_matrix":
        print(f"{k:10s}: {v:.4f}")
print("confusion_matrix:\n", expert_eval['confusion_matrix'])

print("\n--- Induced Rule (Decision Tree) Evaluation (test set) ---")
for k, v in tree_eval.items():
    if k != "confusion_matrix":
        print(f"{k:10s}: {v:.4f}")
print("confusion_matrix:\n", tree_eval['confusion_matrix'])

print("\n--- ROC & AUC ---")
print(f"Expert Rules (approx) AUC: {auc_exp:.4f}")
print(f"Induced Rules (Tree) AUC: {auc_tree:.4f}")

print("\n--- Interpretability proxies ---")
print(f"Expert rules: {expert_num_rules} rules, avg tokens per rule ≈ {expert_avg_tokens:.1f}")
print(f"Induced tree: approx {tree_num_leaf_rules} leaf rules, avg '|' clauses ≈ {tree_avg_clause:.2f}")

# Top triggered expert rules frequency
rule_freq = Counter(expert_rules_used_test)
print("\nTop expert rules triggered in test set:")
for rule, cnt in rule_freq.most_common():
    print(f"  {rule:50s} -> {cnt}")

# ---------------------------
# 8. Visualizations
# ---------------------------
def plot_feature_importance(model, feature_names, out_path=None):
    fi = model.feature_importances_
    idx = np.argsort(fi)[::-1]
    plt.figure(figsize=(10,6))
    plt.bar(range(len(fi)), fi[idx])
    plt.xticks(range(len(fi)), np.array(feature_names)[idx], rotation=90)
    plt.title("Decision Tree Feature Importances")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()

def plot_roc_curves(y_test, expert_probs, tree_probs, auc_exp_val, auc_tree_val, out_path=None):
    fpr_e, tpr_e, _ = roc_curve(y_test, expert_probs)
    fpr_t, tpr_t, _ = roc_curve(y_test, tree_probs)
    plt.figure(figsize=(7,6))
    plt.plot(fpr_e, tpr_e, label=f"Expert (approx) AUC={auc_exp_val:.3f}")
    plt.plot(fpr_t, tpr_t, label=f"Induced Tree AUC={auc_tree_val:.3f}")
    plt.plot([0,1], [0,1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison")
    plt.legend()
    plt.grid(True)
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()

def plot_confusion_matrix(cm, title="Confusion matrix", out_path=None):
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["Low risk (0)", "High risk (1)"])
    plt.yticks(tick_marks, ["Low risk (0)", "High risk (1)"])
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    plt.show()

# Feature importance (tree)
plot_feature_importance(tree_clf, X_train_tree.columns,
                        out_path=os.path.join(OUTPUT_DIR, "feature_importance.png"))

# ROC curves
plot_roc_curves(y_test, expert_probs_approx, tree_probs_test, auc_exp, auc_tree,
                out_path=os.path.join(OUTPUT_DIR, "roc_comparison.png"))

# Confusion matrices
plot_confusion_matrix(expert_eval['confusion_matrix'], title="Expert Rules Confusion Matrix",
                      out_path=os.path.join(OUTPUT_DIR, "cm_expert.png"))
plot_confusion_matrix(tree_eval['confusion_matrix'], title="Induced Tree Confusion Matrix",
                      out_path=os.path.join(OUTPUT_DIR, "cm_tree.png"))

# Expert rule trigger distribution (horizontal bar)
plt.figure(figsize=(8,4))
labels = [r for r, _ in rule_freq.items()]
counts = [c for _, c in rule_freq.items()]
# sort by counts
sorted_pairs = sorted(zip(counts, labels), reverse=True)
counts_s, labels_s = zip(*sorted_pairs)
plt.barh(range(len(labels_s)), counts_s)
plt.yticks(range(len(labels_s)), labels_s)
plt.title("Expert Rule Trigger Frequency (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "expert_rule_freq.png"), dpi=150)
plt.show()

# ---------------------------
# 9. Export artifacts (data + rules + short summary)
# ---------------------------
sample_csv_path = os.path.join(OUTPUT_DIR, "credit_data_sample_200.csv")
df.sample(200, random_state=RANDOM_STATE).to_csv(sample_csv_path, index=False)
rules_txt_path = os.path.join(OUTPUT_DIR, "decision_tree_rules.txt")
with open(rules_txt_path, "w") as f:
    f.write(tree_rules_text)

summary_txt_path = os.path.join(OUTPUT_DIR, "summary_evaluation.txt")
with open(summary_txt_path, "w") as f:
    f.write("Expert Rule System Evaluation (test set):\n")
    for k, v in expert_eval.items():
        f.write(f"{k}: {v}\n")
    f.write("\nInduced Rule (Decision Tree) Evaluation (test set):\n")
    for k, v in tree_eval.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nExpert AUC (approx): {auc_exp}\n")
    f.write(f"Tree AUC: {auc_tree}\n")
    f.write("\nExtracted decision tree rules:\n")
    f.write(tree_rules_text)

print("\nSaved sample CSV to:", sample_csv_path)
print("Saved tree rules to:", rules_txt_path)
print("Saved summary to:", summary_txt_path)
print("Saved plots & images into:", OUTPUT_DIR)

# ---------------------------
# 10. Helper: classify a new applicant with both systems
# ---------------------------
def classify_new_applicant(applicant_dict):
    """
    applicant_dict: dict with keys matching X columns:
      ['age','annual_income','credit_amount','duration_months','employment_years',
       'previous_defaults','num_dependents','purpose','debt_to_income']
    returns: dict with expert prediction/rule and induced-rule prediction/probability
    """
    r = pd.Series(applicant_dict)
    expert_p, expert_rule = expert_rule_predict_one(r)
    # Prepare for tree (one-hot encode purpose and align columns)
    r_df = r.to_frame().T.copy()
    r_df = pd.get_dummies(r_df, columns=['purpose'], drop_first=True)
    for c in X_train_tree.columns:
        if c not in r_df.columns:
            r_df[c] = 0
    r_df = r_df[X_train_tree.columns]
    tree_p = int(tree_clf.predict(r_df)[0])
    tree_prob = float(tree_clf.predict_proba(r_df)[0,1])
    return {"expert_pred": int(expert_p), "expert_rule": expert_rule,
            "tree_pred": tree_p, "tree_prob": tree_prob}

# Example usage of classifier
example_applicant = {
    'age': 34,
    'annual_income': 22000,
    'credit_amount': 18000,
    'duration_months': 48,
    'employment_years': 1,
    'previous_defaults': 0,
    'num_dependents': 1,
    'purpose': 'car',
    'debt_to_income': round(18000/(22000+1), 4)
}
print("\nExample new applicant:", example_applicant)
print("Classification:", classify_new_applicant(example_applicant))

# ---------------------------
# 11. If desired: print the first 200 lines of the extracted rule text to console (optional)
# ---------------------------
print("\n--- Extracted Decision Tree Rules (first 1200 chars) ---\n")
print(tree_rules_text[:1200].rstrip())
