import io
import os
import joblib
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from common import read_uploaded_csv, prepare_features_att
from common_app import tokens_transformer

APP_MODEL_PATH = os.getenv("APP_MODEL_PATH", "/srv/models/app_model.joblib")
ATT_MODEL_PATH = os.getenv("ATT_MODEL_PATH", "/srv/models/att_model.joblib")

APP_TRAIN_PATH = os.getenv("APP_TRAIN_PATH", "/srv/data/radcom_app_train.csv")
ATT_TRAIN_PATH = os.getenv("ATT_TRAIN_PATH", "/srv/data/radcom_att_train.csv")

MODELS_DIR = os.path.dirname(APP_MODEL_PATH) or "/srv/models"
os.makedirs(MODELS_DIR, exist_ok=True)



ONEHOT_COLS = ["Protocol", "Destination_port"]

HASH_INPUT_COLS = [
    "Protocol", "Destination_IP", "Destination_port",
    "dst_socket", "Source_IP", "Source_port", "src_socket"
]

FULL_COLS = list(dict.fromkeys(ONEHOT_COLS + HASH_INPUT_COLS))



LABEL2ID = {
    "vod": 0,
    "file download": 1,
    "real_time_audio": 2,
    "real_time_messaging": 3,
    "real_time_video": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}



app = FastAPI(title="Traffic Inference API", version="1.0.0")

app_model = None
att_model = None



def _csv_response(df: pd.DataFrame, out_name: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )


def _ensure_app_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    must = ["Protocol", "Source_IP", "Destination_IP", "Source_port", "Destination_port"]
    for c in must:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing required column: {c}")
        df[c] = df[c].astype(str).str.lower()

    df["dst_socket"] = df["Destination_IP"].astype(str) + ":" + df["Destination_port"].astype(str)
    df["src_socket"] = df["Source_IP"].astype(str) + ":" + df["Source_port"].astype(str)
    return df


def train_app_if_needed():
    global app_model

    if os.path.exists(APP_MODEL_PATH):
        app_model = joblib.load(APP_MODEL_PATH)
        return

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
    from sklearn.feature_extraction import FeatureHasher
    from sklearn.svm import LinearSVC

    if not os.path.exists(APP_TRAIN_PATH):
        raise RuntimeError(f"APP train file not found at: {APP_TRAIN_PATH}")

    train = pd.read_csv(APP_TRAIN_PATH)
    train.columns = train.columns.str.strip()


    label_col = None
    for c in ["label", "app", "application", "target", "y"]:
        if c in train.columns:
            label_col = c
            break
    if label_col is None:
        raise RuntimeError("APP train file has no label column (label/app/...)")

    train = _ensure_app_cols(train)

    X_train = train[FULL_COLS]
    y_train = train[label_col].astype(str)

    pre = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
        ("hash", Pipeline([
            ("to_tokens", FunctionTransformer(tokens_transformer, validate=False)),
            ("hasher", FeatureHasher(
                n_features=2**22,
                input_type="string",
                alternate_sign=False
            )),
        ]), HASH_INPUT_COLS),
    ])

    app_model = Pipeline([
        ("pre", pre),
        ("clf", LinearSVC(C=0.7, max_iter=80000)),
    ]).fit(X_train, y_train)

    joblib.dump(app_model, APP_MODEL_PATH)


def train_att_if_needed():
    global att_model

    if os.path.exists(ATT_MODEL_PATH):
        att_model = joblib.load(ATT_MODEL_PATH)
        return

    from sklearn.ensemble import RandomForestClassifier

    if not os.path.exists(ATT_TRAIN_PATH):
        raise RuntimeError(f"ATT train file not found at: {ATT_TRAIN_PATH}")

    df = pd.read_csv(ATT_TRAIN_PATH)
    df.columns = df.columns.str.strip()

    if "attribution" not in df.columns:
        raise RuntimeError("ATT train file missing 'attribution' column")

    df["attribution"] = df["attribution"].astype(str).str.strip().str.lower()
    y = df["attribution"].map(LABEL2ID)

    mask = y.notna()
    df = df.loc[mask].copy()
    y = y.loc[mask].astype(int)

    X = prepare_features_att(df)

    att_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    att_model.fit(X, y)

    joblib.dump(att_model, ATT_MODEL_PATH)



@app.on_event("startup")
def load_or_train_models():
    try:
        train_app_if_needed()
    except Exception as e:
        print(f"[APP MODEL] Failed: {e}")

    try:
        train_att_if_needed()
    except Exception as e:
        print(f"[ATT MODEL] Failed: {e}")



@app.get("/health")
def health():
    return {
        "status": "ok",
        "app_model_loaded": app_model is not None,
        "att_model_loaded": att_model is not None,
        "app_model_path": APP_MODEL_PATH,
        "att_model_path": ATT_MODEL_PATH,
        "app_train_path": APP_TRAIN_PATH,
        "att_train_path": ATT_TRAIN_PATH,
    }


@app.post("/predict/app")
def predict_app(file: UploadFile = File(...)):
    if app_model is None:
        raise HTTPException(status_code=503, detail="App model not loaded/trained")

    raw = file.file.read()
    df = read_uploaded_csv(raw)
    df = _ensure_app_cols(df)

    X = df[FULL_COLS]
    pred = app_model.predict(X)

    out = df.copy()
    out["prediction"] = pred
    return _csv_response(out, "val_with_prediction_app.csv")


@app.post("/predict/att")
def predict_att(file: UploadFile = File(...)):
    if att_model is None:
        raise HTTPException(status_code=503, detail="Att model not loaded/trained")

    raw = file.file.read()
    df = read_uploaded_csv(raw)

    X = prepare_features_att(df)


    if hasattr(att_model, "feature_names_in_"):
        X = X.reindex(columns=list(att_model.feature_names_in_), fill_value=0)

    pred_ids = att_model.predict(X)
    pred = [ID2LABEL.get(int(x), "unknown") for x in pred_ids]

    out = df.copy()
    out["prediction"] = pred
    return _csv_response(out, "val_with_prediction_att.csv")
