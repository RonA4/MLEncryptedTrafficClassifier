import io
import pandas as pd


def read_uploaded_csv(upload_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(upload_bytes))
    df.columns = df.columns.str.strip()
    return df


def prepare_features_att(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    drop_cols = [
        "Source_IP", "Destination_IP", "Source_port", "Destination_port",
        "Timestamp", "Flow_ID", "attribution"
    ]

    # avg packet size
    if "Octets" in df.columns and "Packets" in df.columns:
        df["avg_pkt_size"] = df["Octets"] / df["Packets"].replace(0, 1)


    if "Duration" in df.columns and "Octets" in df.columns:
        df["bytes_per_sec"] = df["Octets"] / df["Duration"].replace(0, 0.001)

    if "Duration" in df.columns and "Packets" in df.columns:
        df["pkts_per_sec"] = df["Packets"] / df["Duration"].replace(0, 0.001)

    # protocol mapping
    if "Protocol" in df.columns:
        df["Protocol"] = (
            df["Protocol"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"TCP": 0, "UDP": 1})
            .fillna(-1)
        )

    cols_to_drop = [c for c in drop_cols if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)


    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df
