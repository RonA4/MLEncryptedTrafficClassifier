import argparse
import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction import FeatureHasher
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    f1_score,
)
import joblib


from common_app import tokens_transformer, HASH_INPUT_COLS


TRAIN = "APP-1/radcom_app_train.csv"
TEST  = "APP-1/radcom_app_test.csv"


def pick_label_col(df):
    for c in ["label", "app", "app_id", "application", "target", "y"]:
        if c in df.columns:
            return c
    raise ValueError("not found-  label")


def unique_cols(cols):
    out = []
    for c in cols:
        if c not in out:
            out.append(c)
    return out

ONEHOT_COLS = ["Protocol", "Destination_port"]
FULL_COLS = unique_cols(ONEHOT_COLS + HASH_INPUT_COLS)



def topk_accuracy_from_scores(y_true, scores, classes, k):
    topk_idx = np.argpartition(-scores, kth=min(k, scores.shape[1]-1), axis=1)[:, :k]
    topk_labels = classes[topk_idx]
    return np.mean((topk_labels == y_true[:, None]).any(axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_features", type=int, default=2**22)
    ap.add_argument("--max_iter", type=int, default=80000)
    ap.add_argument("--print_labels", action="store_true")
    ap.add_argument("--out_path", default="traffic-system/models/app_model.joblib")
    args = ap.parse_args()

    train = pd.read_csv(TRAIN)
    test  = pd.read_csv(TEST)

    train.columns = train.columns.str.strip()
    test.columns  = test.columns.str.strip()

    label_col = pick_label_col(train)

    must = ["Protocol", "Source_IP", "Destination_IP", "Source_port", "Destination_port"]
    for c in must:
        train[c] = train[c].astype(str).str.lower()
        test[c]  = test[c].astype(str).str.lower()


    train["dst_socket"] = train["Destination_IP"].astype(str) + ":" + train["Destination_port"].astype(str)
    test["dst_socket"]  = test["Destination_IP"].astype(str) + ":" + test["Destination_port"].astype(str)
    train["src_socket"] = train["Source_IP"].astype(str) + ":" + train["Source_port"].astype(str)
    test["src_socket"]  = test["Source_IP"].astype(str) + ":" + test["Source_port"].astype(str)

    X_train = train[FULL_COLS]
    y_train = train[label_col].astype(str)
    X_test  = test[FULL_COLS]
    y_test  = test[label_col].astype(str)

    pre = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
        ("hash", Pipeline([
            ("to_tokens", FunctionTransformer(tokens_transformer, validate=False)),
            ("hasher", FeatureHasher(
                n_features=args.n_features,
                input_type="string",
                alternate_sign=False
            )),
        ]), HASH_INPUT_COLS)
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", LinearSVC(C=0.7, max_iter=args.max_iter)),
    ]).fit(X_train, y_train)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    joblib.dump(model, args.out_path)
    print(f"✅ Saved model to: {args.out_path}")

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")
    weighted_f1 = f1_score(y_test, pred, average="weighted")

    print("\n=== TEST EVALUATION ===")
    print(f"Top-1 Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")

    labels = np.unique(np.concatenate([y_train, y_test]))
    cm = confusion_matrix(y_test, pred, labels=labels)

    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        cm,
        index=[f"true:{l}" for l in labels],
        columns=[f"pred:{l}" for l in labels]
    ))

    print("\nClassification Report:")
    print(classification_report(y_test, pred, digits=4, zero_division=0))

    scores = model.decision_function(X_test)
    if scores.ndim == 2:
        classes = model.named_steps["clf"].classes_
        print(f"Top-3 Accuracy: {topk_accuracy_from_scores(y_test.values, scores, classes, 3):.4f}")
        print(f"Top-5 Accuracy: {topk_accuracy_from_scores(y_test.values, scores, classes, 5):.4f}")

    if args.print_labels:
        print("\nLabel order:")
        for i, l in enumerate(labels):
            print(i, l)


if __name__ == "__main__":
    main()
