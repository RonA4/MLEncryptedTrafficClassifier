import ipaddress
import pandas as pd


HASH_INPUT_COLS = [
    "Protocol", "Destination_IP", "Destination_port",
    "dst_socket", "Source_IP", "Source_port", "src_socket"
]


def safe_ip_info(ip_str: str):
    try:
        ip_obj = ipaddress.ip_address(str(ip_str).strip())
    except Exception:
        return None, None, None, None, None

    is_v6 = int(ip_obj.version == 6)
    flags = {
        "private": int(ip_obj.is_private),
        "loopback": int(ip_obj.is_loopback),
        "multicast": int(ip_obj.is_multicast),
        "link_local": int(ip_obj.is_link_local),
        "reserved": int(ip_obj.is_reserved),
        "unspecified": int(ip_obj.is_unspecified),
    }

    ip24 = ip16 = ip8 = None
    if ip_obj.version == 4:
        parts = str(ip_obj).split(".")
        if len(parts) == 4:
            ip24 = ".".join(parts[:3])
            ip16 = ".".join(parts[:2])
            ip8 = parts[0]

    return ip24, ip16, ip8, flags, is_v6


def port_bucket(port_str: str) -> str:
    try:
        p = int(str(port_str).strip())
    except Exception:
        return "unknown"
    if 0 <= p <= 1023:
        return "well_known"
    if 1024 <= p <= 49151:
        return "registered"
    if 49152 <= p <= 65535:
        return "ephemeral"
    return "unknown"


def port_mod_bucket(port_str: str) -> str:
    try:
        p = int(str(port_str).strip())
    except Exception:
        return "na"
    if p < 0:
        return "na"
    return str(p % 100)


def build_tokens_fast(df: pd.DataFrame, base_cols):
    out = []
    for r in df.itertuples(index=False):
        row = r._asdict()
        toks = []

        for c in base_cols:
            toks.append(f"{c}={row.get(c)}")

        proto = row.get("Protocol")
        dst_socket = row.get("dst_socket")
        src_socket = row.get("src_socket")
        dst_ip = row.get("Destination_IP")
        src_ip = row.get("Source_IP")
        dst_port = row.get("Destination_port")
        src_port = row.get("Source_port")

        dst24, dst16, dst8, _, _ = safe_ip_info(dst_ip)
        src24, src16, src8, _, _ = safe_ip_info(src_ip)

        if dst24: toks.append(f"dst_/24={dst24}")
        if dst16: toks.append(f"dst_/16={dst16}")
        if dst8:  toks.append(f"dst_/8={dst8}")

        if src24: toks.append(f"src_/24={src24}")
        if src16: toks.append(f"src_/16={src16}")
        if src8:  toks.append(f"src_/8={src8}")

        toks.append(f"dst_port_bucket={port_bucket(dst_port)}")
        toks.append(f"src_port_bucket={port_bucket(src_port)}")
        toks.append(f"dst_port_mod100={port_mod_bucket(dst_port)}")
        toks.append(f"src_port_mod100={port_mod_bucket(src_port)}")

        if proto and dst_socket:
            toks.append(f"proto|dst_socket={proto}|{dst_socket}")
        if src_socket and dst_socket:
            toks.append(f"src_socket|dst_socket={src_socket}|{dst_socket}")

        out.append(toks)

    return out


def tokens_transformer(X):
    df = pd.DataFrame(X, columns=HASH_INPUT_COLS)
    return build_tokens_fast(df, HASH_INPUT_COLS)
