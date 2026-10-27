"""
Vial Temperature Tools - local Streamlit app

Tab 1: view / export a time window of a single CSV file
Tab 2: compare several CSV files, aligned on their lowest temperature

Run with:
    pip install streamlit pandas numpy scipy matplotlib
    streamlit run vial_temp_app.py
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import savgol_filter

st.set_page_config(page_title="Vial Temperature Tools", layout="wide")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "lines.antialiased": True,
    "text.antialiased": True,
})

REQUIRED_COLUMNS = ["Value", "LocalTime", "DeviceDescription"]
EXPORT_COLUMNS = ["LocalTime", "ElapsedTime_min", "Value", "Timestamp",
                  "DeviceData_Id", "DeviceDescription"]
DISPLAY_COLUMNS = ["LocalTime", "ElapsedTime_min", "Value",
                   "DeviceData_Id", "DeviceDescription"]


# ============================================================
# SOURCES (uploaded files or a local folder)
# ============================================================

@dataclass
class Source:
    name: str
    path: Optional[Path] = None
    data: Optional[bytes] = None


# ============================================================
# DATA LOADING
# ============================================================

def convert_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", dayfirst=True)


@st.cache_data(show_spinner=False)
def _load_vial(device: str, path_str: Optional[str], mtime: Optional[float],
               data: Optional[bytes]) -> pd.DataFrame:
    # mtime is only used so the cache refreshes when a file changes on disk
    src = io.BytesIO(data) if data is not None else path_str
    df = pd.read_csv(src, sep=",", low_memory=False)
    df.columns = df.columns.str.strip()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {df.columns.tolist()}")

    vial = df[
        df["DeviceDescription"].astype(str).str.strip().str.casefold()
        == device.strip().casefold()
    ].copy()

    vial["Value"] = pd.to_numeric(vial["Value"], errors="coerce")
    vial["LocalTime"] = convert_datetime(vial["LocalTime"])

    vial = (
        vial.dropna(subset=["Value", "LocalTime"])
        .sort_values("LocalTime")
        .reset_index(drop=True)
    )
    if vial.empty:
        raise ValueError(f"No valid '{device}' measurements found.")
    return vial


def load_vial(source: Source, device: str) -> pd.DataFrame:
    if source.data is not None:
        return _load_vial(device, None, None, source.data)
    return _load_vial(device, str(source.path), source.path.stat().st_mtime, None)


def align_to_minimum(vial: pd.DataFrame, name: str, occurrence: str) -> pd.DataFrame:
    d = vial.copy()
    min_temp = d["Value"].min()
    rows = d[d["Value"] == min_temp]
    idx = rows.index[-1] if occurrence == "last" else rows.index[0]
    min_time = d.loc[idx, "LocalTime"]

    d["RelativeTime_min"] = (d["LocalTime"] - min_time).dt.total_seconds() / 60
    d["SourceFile"] = name
    d["MinimumTemperature_K"] = min_temp
    d["MinimumTemperatureTime"] = min_time
    return d


def sanitize_filename(name: str, strip_exts=(".csv",)) -> str:
    name = name.strip()
    for ext in strip_exts:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
    for ch in '<>:"/\\|?*':
        name = name.replace(ch, "_")
    return name


def smooth_signal(values, method: str, window: int, order: int) -> np.ndarray:
    """Return smoothed values; never modifies original measurements."""
    y = np.asarray(values, dtype=float)
    if method == "none" or len(y) < 3:
        return y.copy()

    window = min(int(window), len(y))
    if method == "savgol":
        if window % 2 == 0:
            window -= 1
        if window <= order or window < 3:
            return y.copy()
        return savgol_filter(y, window_length=window, polyorder=order, mode="interp")
    if method == "mean":
        return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy()
    if method == "median":
        return pd.Series(y).rolling(window, center=True, min_periods=1).median().to_numpy()
    return y.copy()


# ============================================================
# SIDEBAR: SOURCE + SETTINGS
# ============================================================

st.title("Vial Temperature Tools")

folder: Optional[Path] = None
sources: list[Source] = []

with st.sidebar:
    st.header("Data source")
    mode = st.radio("Source", ["Upload files", "Local folder path"], horizontal=True)

    if mode == "Upload files":
        uploads = st.file_uploader("CSV files", type=["csv"], accept_multiple_files=True)
        sources = [Source(u.name, data=u.getvalue()) for u in uploads or []]
    else:
        folder_text = st.text_input("Full path to folder", placeholder=r"C:\data\experiments")
        if folder_text.strip():
            p = Path(folder_text.strip().strip('"'))
            if p.is_dir():
                folder = p
                csvs = sorted(
                    (f for f in p.iterdir() if f.is_file() and f.suffix.lower() == ".csv"),
                    key=lambda f: f.name.lower(),
                )
                sources = [Source(f.name, path=f) for f in csvs]
                st.caption(f"{len(sources)} CSV file(s) found")
            else:
                st.error("Folder not found.")

    st.header("Settings")
    device_description = st.text_input("Device description", value="Vial Temperature (K)")
    minutes_before_min = st.number_input("Minutes before minimum (auto X min)", value=30, step=5)
    min_occurrence = st.selectbox("Minimum occurrence", ["first", "last"])

if not sources:
    st.info("Add CSV files (or enter a folder path) in the sidebar to get started.")
    st.stop()

source_by_name = {s.name: s for s in sources}
names = list(source_by_name)

tab_single, tab_aligned = st.tabs(["Single file: time window", "Compare files: aligned profiles"])


# ============================================================
# TAB 1 - SINGLE FILE TIME WINDOW
# ============================================================

with tab_single:
    chosen_name = st.selectbox("CSV file", names, key="single_file")
    source = source_by_name[chosen_name]

    try:
        vial_temp = load_vial(source, device_description)
    except Exception as err:
        st.error(f"Error: {err}")
        vial_temp = None

    if vial_temp is not None:
        min_dt = vial_temp["LocalTime"].min()
        max_dt = vial_temp["LocalTime"].max()

        c1, c2, c3 = st.columns(3)
        c1.metric("Measurements found", f"{len(vial_temp):,}")
        c2.metric("First timestamp", min_dt.strftime("%d-%m-%Y %H:%M:%S"))
        c3.metric("Last timestamp", max_dt.strftime("%d-%m-%Y %H:%M:%S"))

        st.subheader("Select the data window")
        available_dates = sorted(vial_temp["LocalTime"].dt.date.unique())

        col_date, col_start, col_end = st.columns(3)
        selected_date = col_date.selectbox(
            "Date", options=available_dates,
            format_func=lambda d: pd.Timestamp(d).strftime("%d-%m-%Y"),
            key=f"date_{chosen_name}",
        )

        day_data = vial_temp[vial_temp["LocalTime"].dt.date == selected_date]
        first_time = day_data["LocalTime"].min().time().replace(microsecond=0)
        last_time = day_data["LocalTime"].max().time().replace(microsecond=0)

        # Keys include file + date so defaults reset when either changes
        start_time = col_start.time_input(
            "Start", value=first_time, step=1, key=f"start_{chosen_name}_{selected_date}")
        end_time = col_end.time_input(
            "End", value=last_time, step=1, key=f"end_{chosen_name}_{selected_date}")

        start_dt = pd.Timestamp.combine(selected_date, start_time)
        end_dt = pd.Timestamp.combine(selected_date, end_time)

        if end_dt <= start_dt:
            st.error("The end time must be later than the start time.")
        else:
            shown = vial_temp[
                (vial_temp["LocalTime"] >= start_dt) & (vial_temp["LocalTime"] <= end_dt)
            ].copy()

            if shown.empty:
                st.warning("No measurements were found in the selected time window.")
            else:
                shown["ElapsedTime_min"] = (
                    shown["LocalTime"] - shown["LocalTime"].min()
                ).dt.total_seconds() / 60

                fig, ax = plt.subplots(figsize=(13, 6))
                ax.plot(shown["LocalTime"], shown["Value"], color="#1565C0", linewidth=1.7)
                ax.set_title("Vial Temperature Profile", fontsize=15, fontweight="bold")
                ax.set_xlabel("Local time", fontsize=12)
                ax.set_ylabel("Vial Temperature (K)", fontsize=12)
                ax.grid(True, linestyle="--", alpha=0.25)
                locator = mdates.AutoDateLocator(minticks=4, maxticks=12)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                ax.set_xlim(start_dt, end_dt)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                duration = (end_dt - start_dt).total_seconds() / 60
                st.write(
                    f"**Selected window:** {start_dt:%d-%m-%Y %H:%M:%S} to {end_dt:%H:%M:%S}  \n"
                    f"**Window duration:** {duration:.2f} minutes  \n"
                    f"**Measurements shown:** {len(shown):,}  \n"
                    f"**Temperature range:** {shown['Value'].min():.2f}–{shown['Value'].max():.2f} K"
                )

                with st.expander("Preview (first 20 rows)", expanded=True):
                    st.dataframe(
                        shown[[c for c in DISPLAY_COLUMNS if c in shown.columns]].head(20),
                        use_container_width=True,
                    )

                st.subheader("Export the data shown in the graph")
                fname = sanitize_filename(
                    st.text_input("File name", value="vial_temperature_selected_window",
                                  key="single_export_name")
                )
                export_df = shown[[c for c in EXPORT_COLUMNS if c in shown.columns]]
                csv_bytes = export_df.to_csv(
                    index=False, sep=",", date_format="%Y-%m-%d %H:%M:%S.%f"
                ).encode("utf-8")

                b1, b2 = st.columns(2)
                b1.download_button("Download CSV", csv_bytes,
                                   file_name=f"{fname or 'export'}.csv", mime="text/csv",
                                   type="primary", disabled=not fname)
                if source.path is not None and b2.button(
                        "Save next to original file", disabled=not fname):
                    out = source.path.parent / f"{fname}.csv"
                    out.write_bytes(csv_bytes)
                    st.success(f"Export completed: {out}  ({len(export_df):,} rows)")


# ============================================================
# TAB 2 - ALIGNED COMPARISON
# ============================================================

def compute_auto_limits(datasets, minutes_before):
    x_max = max(d["RelativeTime_min"].max() for d in datasets)
    values = pd.concat(
        [d.loc[d["RelativeTime_min"] >= -minutes_before, "Value"] for d in datasets],
        ignore_index=True,
    ).dropna()
    if values.empty:
        return None
    lo, hi = values.min(), values.max()
    margin = (hi - lo) * 0.05 if hi > lo else 1.0
    return (float(-minutes_before), float(max(x_max, 1)),
            float(lo - margin), float(hi + margin))


def apply_auto_limits():
    lim = st.session_state.get("aligned_auto")
    if lim:
        (st.session_state["x_min"], st.session_state["x_max"],
         st.session_state["y_min"], st.session_state["y_max"]) = lim


def select_all_files():
    st.session_state["aligned_files"] = list(names)


def clear_files():
    st.session_state["aligned_files"] = []


with tab_aligned:
    st.subheader("1. Select CSV files")

    # Drop stale selections (e.g. after changing folder)
    st.session_state["aligned_files"] = [
        n for n in st.session_state.get("aligned_files", []) if n in source_by_name
    ]
    st.multiselect("CSV files", names, key="aligned_files")
    bc1, bc2, _ = st.columns([1, 1, 4])
    bc1.button("Select all", on_click=select_all_files)
    bc2.button("Clear selection", on_click=clear_files)

    selected_names = st.session_state["aligned_files"]
    if not selected_names:
        st.info("Select one or more CSV files.")
        st.stop()

    # ---- load + align
    aligned = []
    results = []
    for n in selected_names:
        try:
            d = align_to_minimum(load_vial(source_by_name[n], device_description), n, min_occurrence)
            aligned.append(d)
            results.append({"File": n, "Status": "Loaded", "Measurements": len(d),
                            "Minimum temperature (K)": d["MinimumTemperature_K"].iloc[0],
                            "Minimum time": d["MinimumTemperatureTime"].iloc[0]})
        except Exception as err:
            results.append({"File": n, "Status": f"Error: {err}", "Measurements": 0,
                            "Minimum temperature (K)": None, "Minimum time": None})

    with st.expander("Loaded files", expanded=len(aligned) < len(selected_names)):
        st.dataframe(pd.DataFrame(results), use_container_width=True)

    if not aligned:
        st.stop()

    # ---- axis limits: reset to automatic whenever the file set / settings change
    auto = compute_auto_limits(aligned, minutes_before_min)
    st.session_state["aligned_auto"] = auto
    signature = (tuple(selected_names), device_description, min_occurrence, minutes_before_min)
    if st.session_state.get("aligned_sig") != signature:
        st.session_state["aligned_sig"] = signature
        if auto:
            apply_auto_limits()
    for key, default in (("x_min", -30.0), ("x_max", 180.0), ("y_min", 200.0), ("y_max", 320.0)):
        st.session_state.setdefault(key, default)

    st.subheader("2. Adjust plot settings")
    a1, a2, a3, a4, a5 = st.columns([1, 1, 1, 1, 1.2])
    a1.number_input("X minimum", key="x_min", step=1.0)
    a2.number_input("X maximum", key="x_max", step=1.0)
    a3.number_input("Y minimum", key="y_min", step=1.0)
    a4.number_input("Y maximum", key="y_max", step=1.0)
    a5.write("")
    a5.button("Automatic limits", on_click=apply_auto_limits)

    line_width = st.slider("Line width", 0.5, 4.0, 1.7, 0.1)

    st.markdown("**Smoothing options**")
    s1, s2, s3, s4 = st.columns(4)
    method_label = s1.selectbox(
        "Smoothing",
        ["None (raw)", "Moving average", "Savitzky-Golay", "Rolling median"])
    method = {"None (raw)": "none", "Moving average": "mean",
              "Savitzky-Golay": "savgol", "Rolling median": "median"}[method_label]
    window = s2.number_input("Window (points)", min_value=1, max_value=10001, value=11, step=2)
    poly_order = s3.number_input("Poly. order", min_value=0, max_value=10, value=2)
    show_raw = s4.checkbox("Overlay raw signal")

    o1, o2, o3 = st.columns(3)
    show_grid = o1.checkbox("Show grid", value=True)
    show_min_line = o2.checkbox("Show minimum line", value=True)
    show_shading = o3.checkbox("Shade time before minimum", value=False)

    x_min, x_max = st.session_state["x_min"], st.session_state["x_max"]
    y_min, y_max = st.session_state["y_min"], st.session_state["y_max"]

    if x_min >= x_max:
        st.error("X minimum must be lower than X maximum.")
        st.stop()
    if y_min >= y_max:
        st.error("Y minimum must be lower than Y maximum.")
        st.stop()

    # ---- plot
    fig, ax = plt.subplots(figsize=(16, 7))
    cmap = plt.get_cmap("tab20", max(len(aligned), 1))
    plotted_files = 0
    plotted_points = 0

    for i, d in enumerate(aligned):
        mask = ((d["RelativeTime_min"] >= x_min) & (d["RelativeTime_min"] <= x_max)).to_numpy()
        if not mask.any():
            continue
        visible = d[mask]
        plotted_files += 1
        plotted_points += len(visible)

        # Smooth the complete aligned trace BEFORE cropping (avoids edge artifacts)
        smoothed = smooth_signal(d["Value"].to_numpy(), method, window, poly_order)

        if show_raw and method != "none":
            ax.plot(visible["RelativeTime_min"], visible["Value"],
                    color=cmap(i), linewidth=0.7, alpha=0.25, label="_nolegend_")

        ax.plot(visible["RelativeTime_min"], smoothed[mask], color=cmap(i),
                linewidth=line_width, alpha=0.9, label=d["SourceFile"].iloc[0])

    if show_shading and x_min < 0:
        ax.axvspan(x_min, min(0, x_max), color="grey", alpha=0.12, zorder=0)

    if show_min_line and x_min <= 0 <= x_max:
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
                   label="Start Primary Drying")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Aligned Vial-Temperature Profiles", fontsize=16, fontweight="bold")
    ax.set_xlabel("Relative time from lowest temperature (min)", fontsize=12)
    ax.set_ylabel("Vial Temperature (K)", fontsize=12)
    if show_grid:
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if plotted_files > 0:
        ax.legend(title="CSV file", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()

    st.pyplot(fig)
    st.write(
        f"**Displayed files:** {plotted_files}  \n"
        f"**Displayed measurements:** {plotted_points:,}  \n"
        f"**X range:** {x_min:.2f} to {x_max:.2f} min  \n"
        f"**Y range:** {y_min:.2f} to {y_max:.2f} K"
    )

    # ---- JPG export
    st.subheader("3. Export current figure")
    e1, e2, e3 = st.columns([3, 1, 1])
    fig_name = sanitize_filename(
        e1.text_input("Figure name", value="aligned_vial_temperature_profiles"),
        strip_exts=(".jpg", ".jpeg"),
    ) or "aligned_vial_temperature_profiles"
    dpi = e2.selectbox("Resolution (dpi)", [150, 300, 600, 1200], index=1,
                       help="1200 dpi on a 16x7 inch figure is very large and slow to render.")
    e3.write("")
    if e3.button("Generate JPG", type="primary"):
        with st.spinner("Rendering figure..."):
            buf = io.BytesIO()
            fig.savefig(buf, format="jpg", dpi=dpi, bbox_inches="tight", facecolor="white")
            st.session_state["jpg_bytes"] = buf.getvalue()
            st.session_state["jpg_name"] = f"{fig_name}.jpg"

    if st.session_state.get("jpg_bytes"):
        size_mb = len(st.session_state["jpg_bytes"]) / 1e6
        st.caption(f"Generated {st.session_state['jpg_name']} ({size_mb:.1f} MB) "
                   "from the plot as it was when you clicked Generate.")
        d1, d2, _ = st.columns([1, 1.4, 3])
        d1.download_button("Download JPG", st.session_state["jpg_bytes"],
                           file_name=st.session_state["jpg_name"], mime="image/jpeg")
        if folder is not None and d2.button("Save in selected folder"):
            out = folder / st.session_state["jpg_name"]
            out.write_bytes(st.session_state["jpg_bytes"])
            st.success(f"Figure exported successfully: {out}")

    plt.close(fig)
