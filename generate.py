import os
import sys
import re
import json
import argparse
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union, List
from collections.abc import Iterable as _Iterable

import numpy as np

# Optional: only used if present
try:
    import pandas as _pd  # noqa
except Exception:
    _pd = None

try:
    import wfdb  # for PTB-XL fallback
except Exception:
    wfdb = None

# Make repo's src importable (same convention as original script)
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Import dataset & helper to resolve ECG file path
try:
    from time_series_datasets.ecg_qa.ECGQACoTQADataset import ECGQACoTQADataset
    from time_series_datasets.ecg_qa.plot_example import get_ptbxl_ecg_path
except Exception as e:
    print("ERROR: Could not import ECGQACoTQADataset / helpers from src/ ...")
    print(e)
    sys.exit(1)


# -------------------------- configuration & utilities --------------------------

PROMPT_KEYS = ["pre_prompt", "prompt", "input", "question", "question_text"]
ANSWER_KEYS = ["correct_answer", "answer", "label", "target"]
ECG_ID_KEYS = ["ecg_id", "record_id", "id"]
CONTEXT_KEYS = ["context", "clinical_context", "patient_context", "history"]
TEMPLATE_KEYS = ["template_id", "template", "qid", "question_id"]

LIKELY_CONTAINER_ATTRS = ["samples", "data", "dataset", "examples", "items", "records", "rows"]
LIKELY_DF_ATTRS = ["df", "dataframe", "table"]
LIKELY_SPLIT_METHODS = ["get_split", "load_split", "split", "get", "load", "prepare", "build", "create", "make"]

# Keys where a time-series may live (value may be np.ndarray/list/dict or CSV string)
TS_TOP_LEVEL_KEYS = [
    "time_series", "timeseries", "ts", "ecg", "ecg_data", "signals", "signal",
    "waveform", "lead_data", "lead_signals", "processed_ecg", "processed_time_series",
    "processed_ecg_data", "ptbxl", "ptbxl_100hz", "ten_seconds", "x", "X", "values", "data"
]
# Keys that suggest a CSV string of the time-series
TS_CSV_KEYS = [
    "time_series_csv", "timeseries_csv", "ecg_csv", "signal_csv", "ts_csv",
    "table", "ecg_table", "csv"
]
# Keys that may hold sampling rate
FS_KEYS = ["fs", "sampling_rate", "sampling_frequency", "sr", "hz", "frequency"]


def first_present(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def log_error(log_path: Path, msg: str) -> None:
    try:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


# -------------------------- robust ecg_id parsing --------------------------

def _coerce_int(obj: Any) -> Optional[int]:
    """Coerce scalar to int safely; strings may include digits among other chars."""
    if obj is None:
        return None
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, float) and np.isfinite(obj):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        try:
            val = float(obj)
            if np.isfinite(val):
                return int(val)
        except Exception:
            return None
    if isinstance(obj, str):
        s = obj.strip()
        try:
            return int(s)
        except Exception:
            pass
        m = re.search(r"\d{2,}", s)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
        return None
    return None


def parse_ecg_id(ecg_field: Any) -> Optional[int]:
    """
    Parse ecg_id from many shapes:
    - int/float/str with digits
    - list/tuple (take first element that can coerce to int)
    - dict with keys ecg_id/record_id/id (recursively parsed)
    - nested lists/dicts
    """
    val = _coerce_int(ecg_field)
    if val is not None:
        return val

    if isinstance(ecg_field, dict):
        for k in ("ecg_id", "record_id", "id", "rid"):
            if k in ecg_field:
                v = parse_ecg_id(ecg_field[k])
                if v is not None:
                    return v
        for v in ecg_field.values():
            out = parse_ecg_id(v)
            if out is not None:
                return out
        return None

    if isinstance(ecg_field, (list, tuple, np.ndarray)):
        try:
            iterable = list(ecg_field)
        except Exception:
            iterable = []
        for item in iterable:
            out = parse_ecg_id(item)
            if out is not None:
                return out
        return None

    return None


# -------------------------- CSV detection & creation --------------------------

def prompt_has_embedded_timeseries(prompt_text: Optional[str]) -> bool:
    """
    Return True iff we detect a real CSV timeseries block already embedded, not just the phrase 'time series'.
    Conditions:
      - A header line that starts with 'time_seconds,I,II,III' (case-insensitive), OR
      - At least 5 consecutive CSV-like numeric lines with a consistent column count >= 7.
    """
    if not prompt_text:
        return False

    if re.search(r'(?im)^\s*time[_ ]?seconds\s*,\s*I\s*,\s*II\s*,\s*III', prompt_text):
        return True

    lines = prompt_text.strip().splitlines()
    numeric_run = 0
    prev_cols = None
    for line in lines:
        if re.match(r'^\s*-?\d+(?:\.\d+)?\s*,', line):
            parts = [p.strip() for p in line.split(',')]
            col_count = len(parts)
            if col_count >= 7:
                if prev_cols is None or col_count == prev_cols:
                    numeric_run += 1
                    prev_cols = col_count
                    if numeric_run >= 5:
                        return True
                else:
                    numeric_run = 1
                    prev_cols = col_count
            else:
                numeric_run = 0
                prev_cols = None
        else:
            if numeric_run >= 5:
                return True
            numeric_run = 0
            prev_cols = None

    return False


def is_csv_timeseries(text: str) -> bool:
    """Heuristic to decide if a string looks like our ECG CSV."""
    if not isinstance(text, str) or len(text) < 50:
        return False
    if re.search(r'(?im)^\s*time[_ ]?seconds\s*,', text):
        return True
    # Otherwise check for a decent run of numeric csv lines
    return prompt_has_embedded_timeseries(text)


def timeseries_to_csv_block(sig: np.ndarray, fs: int = 100) -> str:
    """Convert (T x 12) ECG into a compact CSV text block with a time column."""
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    T = sig.shape[0]
    if T <= 0:
        return ""
    t = np.arange(T, dtype=float) / float(fs)
    header = ["time_seconds"] + lead_names
    lines = [",".join(header)]
    for i in range(T):
        vals = []
        for v in sig[i, :12]:
            if isinstance(v, (float, np.floating)) and not np.isfinite(v):
                vals.append("0.00000")
            else:
                vals.append(f"{float(v):.5f}")
        row = [f"{t[i]:.2f}"] + vals
        lines.append(",".join(row))
    return "\n".join(lines)


# -------------------------- find time-series inside sample --------------------------

def _as_numpy(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return None
    # Ensure numeric
    try:
        arr = arr.astype(float)
    except Exception:
        return None
    return arr


def _collect_lead_matrix_from_dict(d: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    If dict looks like {lead_name -> 1D array}, build a (T x L) in lead order.
    """
    if not isinstance(d, dict) or not d:
        return None
    # Prefer canonical ECG lead ordering if possible
    lead_order = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    keys = list(d.keys())
    # Try canonical first
    cols = []
    T = None
    used = set()
    for name in lead_order:
        if name in d:
            arr = _as_numpy(d[name])
            if arr is None or arr.shape[1] != 1:
                return None
            if T is None:
                T = arr.shape[0]
            if arr.shape[0] != T:
                return None
            cols.append(arr[:, 0])
            used.add(name)
    if cols:
        mat = np.column_stack(cols)
        return mat
    # Otherwise, any dict of equal-length 1D arrays (limit max 16 leads)
    try_cols = []
    T = None
    count = 0
    for k, v in d.items():
        arr = _as_numpy(v)
        if arr is None or arr.shape[1] != 1:
            continue
        if T is None:
            T = arr.shape[0]
        if arr.shape[0] != T:
            continue
        try_cols.append(arr[:, 0])
        count += 1
        if count >= 12:
            break
    if try_cols and len(try_cols) >= 6:
        return np.column_stack(try_cols)
    return None


def _search_for_fs(obj: Any) -> Optional[int]:
    """Search for a sampling rate near/with the data."""
    if isinstance(obj, dict):
        for k in FS_KEYS:
            if k in obj:
                v = obj[k]
                try:
                    fv = float(v)
                    if np.isfinite(fv) and fv > 0:
                        return int(round(fv))
                except Exception:
                    pass
        # search shallow in children
        for v in obj.values():
            out = _search_for_fs(v)
            if out is not None:
                return out
    return None


def extract_timeseries_from_sample(sample: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Try very hard to extract a (T x L) numeric matrix and fs from the sample itself.
    Priority:
      1) CSV string fields -> return None matrix but CSV text is handled elsewhere.
      2) Direct 2D arrays under known keys.
      3) Dict of leads -> matrix.
      4) Shallow recursive search in dict/list values.
    Returns (matrix, fs) where matrix may be None if only CSV string exists (handled upstream).
    """
    # 0) If sample already carries a CSV string
    for key in TS_CSV_KEYS:
        if key in sample and isinstance(sample[key], str) and is_csv_timeseries(sample[key]):
            # Indicate "CSV present"; caller should use this text directly.
            return None, _search_for_fs(sample)

    # 1) Direct arrays under known keys
    for key in TS_TOP_LEVEL_KEYS:
        if key in sample:
            val = sample[key]
            # If it's a CSV string, handled by previous block (continue).
            if isinstance(val, str) and is_csv_timeseries(val):
                return None, _search_for_fs(sample)
            # If it's dict of leads
            if isinstance(val, dict):
                mat = _collect_lead_matrix_from_dict(val)
                if mat is not None:
                    fs = _search_for_fs(val) or _search_for_fs(sample)
                    return mat, fs
                # maybe nested dict contains arrays
                for v in val.values():
                    mat2 = _as_numpy(v)
                    if mat2 is not None and mat2.shape[1] >= 6:
                        fs = _search_for_fs(val) or _search_for_fs(sample)
                        return mat2, fs
            # If it's a list/array
            arr = _as_numpy(val)
            if arr is not None and arr.shape[1] >= 6:
                fs = _search_for_fs(sample)
                return arr, fs

    # 2) Shallow recursive search (limited depth for safety)
    def _scan(obj: Any, depth: int = 0) -> Optional[Tuple[np.ndarray, Optional[int]]]:
        if depth > 2:
            return None
        if isinstance(obj, dict):
            # CSV in nested dict?
            for k in TS_CSV_KEYS:
                if k in obj and isinstance(obj[k], str) and is_csv_timeseries(obj[k]):
                    return None, _search_for_fs(obj)
            # arrays in nested dict
            for v in obj.values():
                if isinstance(v, dict):
                    mat = _collect_lead_matrix_from_dict(v)
                    if mat is not None:
                        fs = _search_for_fs(v) or _search_for_fs(obj)
                        return mat, fs
                arr = _as_numpy(v)
                if arr is not None and arr.shape[1] >= 6:
                    fs = _search_for_fs(obj)
                    return arr, fs
            for v in obj.values():
                res = _scan(v, depth + 1)
                if res is not None:
                    return res
        elif isinstance(obj, (list, tuple)):
            for it in obj:
                arr = _as_numpy(it)
                if arr is not None and arr.shape[1] >= 6:
                    return arr, None
                res = _scan(it, depth + 1)
                if res is not None:
                    return res
        return None

    res = _scan(sample, 0)
    if res is not None:
        return res
    return None, None


# -------------------------- resampling & cropping --------------------------

def resample_to_fs(sig: np.ndarray, src_fs: int, dst_fs: int, seconds_limit: Optional[int]) -> np.ndarray:
    """
    Resample 2D array row-wise from src_fs to dst_fs using simple stride or linear interpolation.
    Optionally crop/pad to seconds_limit.
    """
    if sig.ndim != 2:
        raise ValueError("sig must be (T, L)")
    sig = sig.astype(float)
    if src_fs <= 0:
        src_fs = dst_fs

    # Crop first to seconds_limit at src_fs (if provided)
    if seconds_limit is not None and seconds_limit > 0:
        max_len = min(sig.shape[0], seconds_limit * src_fs)
        sig = sig[:max_len]

    if src_fs == dst_fs:
        out = sig
    else:
        factor = src_fs / float(dst_fs)
        if factor >= 1.0 and abs(round(factor) - factor) < 1e-6:
            step = int(round(factor))
            step = max(step, 1)
            out = sig[::step]
        else:
            # Linear interpolation to exact desired length
            desired_len = int(round((sig.shape[0] / src_fs) * dst_fs))
            desired_len = max(desired_len, 1)
            x_old = np.linspace(0.0, 1.0, sig.shape[0])
            x_new = np.linspace(0.0, 1.0, desired_len)
            out = np.empty((desired_len, sig.shape[1]), dtype=float)
            for col in range(sig.shape[1]):
                out[:, col] = np.interp(x_new, x_old, sig[:, col])

    # Now enforce exact seconds_limit at dst_fs (pad or trim)
    if seconds_limit is not None and seconds_limit > 0:
        target_len = seconds_limit * dst_fs
        if out.shape[0] > target_len:
            out = out[:target_len]
        elif out.shape[0] < target_len:
            pad_len = target_len - out.shape[0]
            out = np.vstack([out, np.zeros((pad_len, out.shape[1]), dtype=float)])

    # Ensure 12 leads (pad/truncate)
    if out.shape[1] < 12:
        pad = np.zeros((out.shape[0], 12 - out.shape[1]), dtype=float)
        out = np.concatenate([out, pad], axis=1)
    elif out.shape[1] > 12:
        out = out[:, :12]

    return out


# -------------------------- load from PTB-XL (fallback) --------------------------

def load_ecg_timeseries_from_ptbxl(ecg_id: int, seconds: int = 10, target_fs: int = 100) -> Optional[np.ndarray]:
    if wfdb is None:
        return None
    try:
        base = get_ptbxl_ecg_path(ecg_id)
        if not (os.path.exists(base + ".dat") or os.path.exists(base + ".hea")):
            return None
        sig, meta = wfdb.rdsamp(base)
        fs = int(meta.get("fs", 500))
        if sig.ndim == 1:
            sig = sig[:, None]
        sig = resample_to_fs(sig, src_fs=fs, dst_fs=target_fs, seconds_limit=seconds)
        return sig
    except Exception:
        return None


# -------------------------- build prompt with time-series --------------------------

def build_prompt_with_timeseries(base_prompt: str,
                                 sample: Dict[str, Any],
                                 ecg_id: Optional[int],
                                 embed_policy: str,
                                 ts_seconds: int,
                                 ts_fs: int) -> str:
    """
    Decide whether/how to append time-series, preferring data embedded in the sample.
    Priority:
      1) If prompt already has CSV and embed_policy=='missing' -> keep as-is.
      2) If sample provides CSV string -> append that.
      3) If sample provides arrays -> resample -> append.
      4) Else try PTB-XL via ecg_id.
      5) Else add a note.
    """
    if embed_policy == "never":
        return base_prompt

    already_has_ts = prompt_has_embedded_timeseries(base_prompt)
    if embed_policy == "missing" and already_has_ts:
        return base_prompt

    # 1) CSV string directly in sample?
    for key in TS_CSV_KEYS:
        v = sample.get(key)
        if isinstance(v, str) and is_csv_timeseries(v):
            header = f"\n\n# Time Series (from dataset, CSV)\n"
            return base_prompt + header + v.strip() + "\n"

    # 2) Arrays in sample?
    mat, fs = extract_timeseries_from_sample(sample)
    if mat is not None:
        fs_eff = int(fs) if (isinstance(fs, (int, float)) and fs and fs > 0) else ts_fs
        sig = resample_to_fs(mat, src_fs=fs_eff, dst_fs=ts_fs, seconds_limit=ts_seconds)
        csv_block = timeseries_to_csv_block(sig, fs=ts_fs)
        if csv_block:
            header = f"\n\n# Time Series (from dataset arrays, {ts_fs} Hz, {ts_seconds} s)\n"
            return base_prompt + header + csv_block + "\n"

    # 3) Fallback: PTB-XL files by ecg_id
    if ecg_id is not None:
        sig = load_ecg_timeseries_from_ptbxl(ecg_id, seconds=ts_seconds, target_fs=ts_fs)
        if sig is not None and sig.size > 0:
            csv_block = timeseries_to_csv_block(sig, fs=ts_fs)
            if csv_block:
                header = f"\n\n# Time Series (from PTB-XL, {ts_fs} Hz, {ts_seconds} s)\n"
                return base_prompt + header + csv_block + "\n"

    # 4) Final note if everything else fails
    note_id = f"ecg_id={ecg_id}" if ecg_id is not None else "no ecg_id"
    return base_prompt + f"\n\n[Note] Could not embed time-series from sample or PTB-XL ({note_id})."


# -------------------------- dataset iteration helpers --------------------------

def _normalize_sample(x):
    if isinstance(x, dict):
        return x
    if hasattr(x, "_asdict"):
        return x._asdict()
    if hasattr(x, "__dict__"):
        return {k: v for k, v in vars(x).items() if not k.startswith("_")}
    return {"value": x}


def _iter_from_container(obj):
    """Yield dict-like samples from an iterable / seq / DF-ish object."""
    # pandas DataFrame?
    if _pd is not None and isinstance(obj, _pd.DataFrame):
        for rec in obj.to_dict("records"):
            yield rec
        return

    # A general iterable (but not string/bytes/dict)
    if isinstance(obj, _Iterable) and not isinstance(obj, (str, bytes, dict)):
        for x in obj:
            yield _normalize_sample(x)
        return

    # Sequence-like with __len__ / __getitem__
    if hasattr(obj, "__len__") and hasattr(obj, "__getitem__"):
        try:
            n = len(obj)  # type: ignore
            for i in range(n):
                yield _normalize_sample(obj[i])  # type: ignore
            return
        except Exception:
            pass


def _try_call(m, *args, **kwargs):
    try:
        return m(*args, **kwargs)
    except Exception:
        return None


def _discover_and_iterate(obj):
    """
    Try hard to find samples in a dataset instance (obj).
    Order of attempts:
      0) Treat the object itself as a container (handles __len__/__getitem__).
      1) If the object is directly iterable.
      2) Known container attributes.
      3) Common split/loader methods returning containers or objects with containers.
    """
    # 0) Treat object itself as container
    gen0 = _iter_from_container(obj)
    if gen0 is not None:
        produced = False
        for rec in gen0:
            produced = True
            yield rec
        if produced:
            return

    # 1) __iter__ present
    if hasattr(obj, "__iter__"):
        try:
            it = iter(obj)
            first = next(it)
            yield _normalize_sample(first)
            for x in it:
                yield _normalize_sample(x)
            return
        except StopIteration:
            return
        except Exception:
            pass

    # 2) Known container attributes
    for name in LIKELY_CONTAINER_ATTRS + LIKELY_DF_ATTRS:
        if hasattr(obj, name):
            container = getattr(obj, name)
            gen = _iter_from_container(container)
            if gen is not None:
                produced = False
                for rec in gen:
                    produced = True
                    yield rec
                if produced:
                    return

    # 3) Try common methods that may return a container or another object with one
    for mname in LIKELY_SPLIT_METHODS:
        if hasattr(obj, mname):
            m = getattr(obj, mname)
            for args in [(), ("test",), ("all",), ("eval",), ("validation",), ("val",)]:
                out = _try_call(m, *args)
                if out is None:
                    continue
                gen = _iter_from_container(out)
                if gen is not None:
                    produced = False
                    for rec in gen:
                        produced = True
                        yield rec
                    if produced:
                        return
                # If out has containers inside
                for name in LIKELY_CONTAINER_ATTRS + LIKELY_DF_ATTRS:
                    if hasattr(out, name):
                        gen = _iter_from_container(getattr(out, name))
                        if gen is not None:
                            produced = False
                            for rec in gen:
                                produced = True
                                yield rec
                            if produced:
                                return


def _diagnose_dataset_class(cls):
    members = [n for n, _ in inspect.getmembers(cls) if not n.startswith("_")]
    return {
        "class_name": cls.__name__,
        "ctor_signature": str(inspect.signature(cls.__init__)) if hasattr(cls, "__init__") else "(no __init__)",
        "public_attrs": members[:50],
    }


def iter_dataset_samples(
    split: str,
    eos_token: str,
    limit: Optional[int] = None,
    templates: Optional[Iterable[int]] = None,
    format_sample_str: bool = False,
    time_series_format_function=None,
    preload_processed_data: bool = True,
    exclude_comparison: bool = False,
) -> Iterable[Dict[str, Any]]:
    """
    Robust iterator over ECGQACoTQADataset items. Instantiates with required args,
    then auto-discovers how to iterate samples.
    """
    try:
        ds = ECGQACoTQADataset(
            split=split,
            EOS_TOKEN=eos_token,
            format_sample_str=format_sample_str,
            time_series_format_function=time_series_format_function,
            max_samples=None,
            exclude_comparison=exclude_comparison,
            preload_processed_data=preload_processed_data,
        )
    except Exception as e:
        diag = _diagnose_dataset_class(ECGQACoTQADataset)
        msg = [
            "Could not instantiate ECGQACoTQADataset with provided arguments.",
            f"  split={split!r}, EOS_TOKEN={eos_token!r}, format_sample_str={format_sample_str}, preload_processed_data={preload_processed_data}, exclude_comparison={exclude_comparison}",
            f"Instantiation error: {repr(e)}",
            "Class diagnosis:",
            f"  - class: {diag['class_name']}",
            f"  - __init__ signature: {diag['ctor_signature']}",
            f"  - public members (subset): {', '.join(diag['public_attrs'])}",
        ]
        raise RuntimeError("\n".join(msg))

    produced = 0
    for rec in _discover_and_iterate(ds):
        if templates:
            tid = first_present(rec, TEMPLATE_KEYS)
            if tid is not None and int(tid) not in set(int(x) for x in templates):
                continue
        yield rec
        produced += 1
        if limit and produced >= limit:
            return

    if produced == 0:
        diag = _diagnose_dataset_class(ECGQACoTQADataset)
        msg = [
            "Could not iterate ECGQACoTQADataset. No samples found via common access patterns.",
            "Hints:",
            "  • If your dataset exposes its data under a different attribute, add it to LIKELY_CONTAINER_ATTRS.",
            "  • If it uses a custom split/loader method, add it to LIKELY_SPLIT_METHODS.",
            "Class diagnosis:",
            f"  - class: {diag['class_name']}",
            f"  - __init__ signature: {diag['ctor_signature']}",
            f"  - public members (subset): {', '.join(diag['public_attrs'])}",
        ]
        raise RuntimeError("\n".join(msg))


# -------------------------- main export logic --------------------------

def main():
    parser = argparse.ArgumentParser(description="Export prompts and answers (no predictions needed).")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Root directory where 'prompts/' and 'answers/' folders will be created (default: current dir).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of samples to export (default: all available).")
    parser.add_argument("--templates", type=str, default=None,
                        help="Comma-separated template_ids to include (optional).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing prompt/answer files if they exist.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "validation"],
                        help="Dataset split to load (default: test).")
    parser.add_argument("--eos", type=str, default="<eos>",
                        help="EOS token required by ECGQACoTQADataset constructor (default: '<eos>').")
    parser.add_argument("--format-sample-str", action="store_true",
                        help="Ask the dataset to format a sample string if it supports it (default: off).")
    parser.add_argument("--no-preload", action="store_true",
                        help="Disable preload_processed_data (default: enabled).")
    parser.add_argument("--exclude-comparison", action="store_true",
                        help="Pass exclude_comparison=True to the dataset (default: False).")
    parser.add_argument("--embed", type=str, default="always", choices=["always", "missing", "never"],
                        help="When to embed time-series into the prompt (default: always).")
    parser.add_argument("--ts-seconds", type=int, default=10, help="Length of ECG snippet to embed (seconds).")
    parser.add_argument("--ts-fs", type=int, default=100, help="Sampling rate for embedding (Hz).")
    args = parser.parse_args()

    out_root = Path(args.out_dir).resolve()
    prompts_dir = out_root / "prompts"
    answers_dir = out_root / "answers"
    log_path = out_root / "export_errors.log"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)

    template_filter = None
    if args.templates:
        template_filter = [int(x.strip()) for x in args.templates.split(",") if x.strip()]

    # Iterate samples from the dataset
    try:
        samples = iter_dataset_samples(
            split=args.split,
            eos_token=args.eos,
            limit=args.limit,
            templates=template_filter,
            format_sample_str=args.format_sample_str,
            time_series_format_function=None,
            preload_processed_data=(not args.no_preload),
            exclude_comparison=args.exclude_comparison,
        )
    except Exception as e:
        print("ERROR:", e)
        sys.exit(2)

    exported = 0
    errors = 0

    for idx, sample in enumerate(samples, start=1):
        try:
            # EARLY skip for resume: if files exist and not overwriting, skip heavy work
            prompt_path = prompts_dir / f"sample_{idx:06d}.txt"
            answer_path = answers_dir / f"sample_{idx:06d}.txt"
            if not args.overwrite and prompt_path.exists() and answer_path.exists():
                continue

            prompt_text = first_present(sample, PROMPT_KEYS)
            answer_text = first_present(sample, ANSWER_KEYS)
            ecg_raw = first_present(sample, ECG_ID_KEYS)
            ecg_id = parse_ecg_id(ecg_raw)

            # Minimal synthesis if needed
            if not prompt_text:
                context = first_present(sample, CONTEXT_KEYS)
                question = first_present(sample, ["question", "question_text"])
                prompt_text = ""
                if context:
                    prompt_text += f"Clinical Context:\n{context}\n\n"
                prompt_text += (question or "Question: (not provided)")

            # Embed time-series with strong preference for data inside the sample
            prompt_with_ts = build_prompt_with_timeseries(
                base_prompt=prompt_text,
                sample=sample,
                ecg_id=ecg_id,
                embed_policy=args.embed,
                ts_seconds=max(1, int(args.ts_seconds)),
                ts_fs=max(1, int(args.ts_fs)),
            )

            # Fallback for the answer if missing: try label index + options
            if answer_text is None:
                label_idx = first_present(sample, ["label_idx", "answer_idx", "y"])
                options = first_present(sample, ["options", "answer_options", "choices"])
                if (label_idx is not None) and isinstance(options, (list, tuple)):
                    try:
                        li = int(label_idx)
                        if 0 <= li < len(options):
                            answer_text = str(options[li])
                        else:
                            answer_text = "(answer not available)"
                    except Exception:
                        answer_text = "(answer not available)"
                else:
                    answer_text = "(answer not available)"

            # Write files (parallel names)
            if args.overwrite or not prompt_path.exists():
                prompt_path.write_text(prompt_with_ts, encoding="utf-8", newline="\n")
            if args.overwrite or not answer_path.exists():
                answer_path.write_text(str(answer_text).strip(), encoding="utf-8", newline="\n")
            exported += 1

        except Exception as e:
            errors += 1
            msg = f"[WARN] Failed on sample #{idx}: {e}"
            print(msg)
            log_error(log_path, msg)

    print(f"\nDone. Exported {exported} samples into:\n  {prompts_dir}\n  {answers_dir}")
    if errors:
        print(f"Completed with {errors} warnings (see {log_path.name}).")


if __name__ == "__main__":
    main()
