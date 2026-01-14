import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os




LABEL2ID = {

    "vod": 0,

    "file download": 1,

    "real_time_audio": 2,

    "real_time_messaging": 3,

    "real_time_video": 4,

}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def prepare_features(df: pd.DataFrame):

    df = df.copy()


    drop_cols = ["Source_IP", "Destination_IP", "Source_port", "Destination_port", 

                 "Timestamp", "Flow_ID", "attribution"]


    if 'Octets' in df.columns and 'Packets' in df.columns:

        df['avg_pkt_size'] = df['Octets'] / df['Packets'].replace(0, 1)

    

    if 'Duration' in df.columns:

        df['bytes_per_sec'] = df['Octets'] / df['Duration'].replace(0, 0.001)

        df['pkts_per_sec']  = df['Packets'] / df['Duration'].replace(0, 0.001)



    if "Protocol" in df.columns:

        df["Protocol"] = df["Protocol"].map({"TCP": 0, "UDP": 1}).fillna(-1)




    cols_to_drop = [c for c in drop_cols if c in df.columns]

    df.drop(columns=cols_to_drop, inplace=True)

    

    df.fillna(0, inplace=True)

    return df




def train_full_model(train_path: str):

    print(f"Loading FULL Training Data from: {train_path}")

    df = pd.read_csv(train_path)




    df["attribution"] = df["attribution"].astype(str).str.strip().str.lower()

    y = df["attribution"].map(LABEL2ID)

    mask = y.notna()

    df = df.loc[mask]

    y = y.loc[mask]


    X = prepare_features(df)

    

    print(f" Training on {len(X)} samples (100% of the file)...")



    model = RandomForestClassifier(

        n_estimators=100,

        max_depth=10,

        random_state=42,

        n_jobs=-1

    )



    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/att_model.joblib")

    print(" Model Trained Successfully.\n")

    return model




def test_full_model(model, test_path: str):

    print(f"Loading FULL Test Data from: {test_path}")

    try:

        df = pd.read_csv(test_path)

    except FileNotFoundError:

        print(f"❌ Error: The file '{test_path}' was not found.")

        return



    # ניקוי Labels (כדי שנוכל להשוות לאמת)

    df["attribution"] = df["attribution"].astype(str).str.strip().str.lower()

    y_true = df["attribution"].map(LABEL2ID)


    mask = y_true.notna()

    df = df.loc[mask]

    y_true = y_true.loc[mask]




    X_test = prepare_features(df)

    

    print(f"Testing on {len(X_test)} samples (100% of the file)...")




    y_pred = model.predict(X_test)





    print("\n===== FINAL RESULTS (Full Test File) =====")

    acc = accuracy_score(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")

    

    print("\nClassification Report:")

    print(classification_report(y_true, y_pred, target_names=[ID2LABEL[i] for i in range(len(LABEL2ID))]))

    

    print("Confusion Matrix:")

    print(confusion_matrix(y_true, y_pred))

    print("=============================================\n")





if __name__ == "__main__":
    TRAIN_FILE = "attribution/radcom_att_train.csv"
    TEST_FILE  = "attribution/radcom_att_test.csv"
    model = train_full_model(TRAIN_FILE)
    test_full_model(model, TEST_FILE)
