"""
Forward-step APA — BATCH version (several subjects at once, forward steps only).

New protocol: every step detected after the sync event is a FORWARD step (no backward steps,
no alternation). Each subject = one L5 file + one leg file.

Output: one Excel workbook with
  - "Summary"    : one row per subject (mean and SD of each APA variable, included steps only)
  - one sheet per subject: per-step APA variables + mean / SD / n below the table
  - "Parameters" : settings used in the analysis
  - "Variables"  : what each column means

Run:
    pip install -r requirements.txt
    streamlit run app_apa_lote.py
"""
import io
import re
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt

FWD, IGN = "forward", "ignore"

# ---- Author signature (edit here) ----
AUTHOR = "Bianca Callegari"
AUTHOR_INFO = ""
APP_VERSION = "1.1-batch (Oct 2026)"

APA_VARS = {   # column -> label used in the tables / Excel
    "APA_onset_ms": "APA onset (ms before HO, earliest of ML/AP)",
    "APA_duration_ms": "APA duration (ms, onset → HO)",
    "ML_onset_rel_HO_ms": "ML onset (ms re. HO)",
    "ML_peak_m_s2": "ML peak amplitude (m/s²)",
    "ML_peak_rel_HO_ms": "ML time to peak (ms re. HO)",
    "ML_dv_m_s": "ML Δv (m/s)",
    "AP_onset_rel_HO_ms": "AP onset (ms re. HO)",
    "AP_peak_m_s2": "AP peak amplitude (m/s²)",
    "AP_peak_rel_HO_ms": "AP time to peak (ms re. HO)",
    "AP_dv_m_s": "AP Δv (m/s)",
    "RES_peak_m_s2": "Resultant peak (m/s²)",
    "RES_peak_rel_HO_ms": "Resultant time to peak (ms re. HO)",
    "RES_angle_deg": "Resultant direction at peak (°, 0 = forward, + = stance side)",
}
VAR_HELP = {
    "APA_onset_ms": "Start of the APA (earliest of ML and AP onset), ms relative to heel-off. More negative = earlier.",
    "APA_duration_ms": "Time from APA onset to heel-off (ms). Longer = slower / more cautious preparation.",
    "ML_onset_rel_HO_ms": "Start of the lateral (ML) component, ms relative to heel-off.",
    "ML_peak_m_s2": "Largest L5 acceleration toward the stance leg (re. baseline).",
    "ML_peak_rel_HO_ms": "Time of the ML peak, ms relative to heel-off.",
    "ML_dv_m_s": "ML velocity change from onset to heel-off (area under the acceleration curve).",
    "AP_onset_rel_HO_ms": "Start of the forward (AP) component, ms relative to heel-off.",
    "AP_peak_m_s2": "Largest forward L5 acceleration (re. baseline).",
    "AP_peak_rel_HO_ms": "Time of the AP peak, ms relative to heel-off.",
    "AP_dv_m_s": "AP velocity change from onset to heel-off.",
    "RES_peak_m_s2": "Peak of the horizontal resultant √(ML²+AP²) before heel-off.",
    "RES_peak_rel_HO_ms": "Time of the resultant peak, ms relative to heel-off.",
    "RES_angle_deg": "Direction of the horizontal acceleration at its peak: 0° = forward, 90° = stance side.",
}

# =============================================================================
# Processing core (same algorithms as app_apa.py)
# =============================================================================

def read_acc(file_or_buf):
    """Reads a 'time, X, Y, Z' file (comma, semicolon or tab; decimal comma; time in ms or s)."""
    if hasattr(file_or_buf, "read"):
        if hasattr(file_or_buf, "seek"):
            file_or_buf.seek(0)
        raw = file_or_buf.read()
    else:
        raw = open(file_or_buf, "rb").read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    first = raw.splitlines()[1] if len(raw.splitlines()) > 1 else raw
    sep = ";" if first.count(";") >= 3 else ("\t" if first.count("\t") >= 3 else ",")
    if sep != "," and "," in first and "." not in first:      # decimal comma
        raw = raw.replace(",", ".")
    d = pd.read_csv(io.StringIO(raw), sep=sep, skipinitialspace=True, engine="python")
    if d.shape[1] < 4:
        raise ValueError(f"Expected 4 columns (time, X, Y, Z); found {d.shape[1]}.")
    d = d.iloc[:, :4]
    d.columns = ["t", "x", "y", "z"]
    d = d.apply(pd.to_numeric, errors="coerce").dropna()
    d = d.sort_values("t").drop_duplicates("t")
    dt = np.median(np.diff(d["t"].values))
    d["t"] = d["t"] / (1.0 if dt < 0.5 else 1000.0)
    return d.reset_index(drop=True)


def lowpass(s, fc, fs, order=4):
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, s)


def has_gravity(d):
    return float(np.median(np.sqrt(d.x ** 2 + d.y ** 2 + d.z ** 2))) > 4.0


def sampling_info(d):
    dt = np.diff(d.t.values) * 1000
    return dict(n=len(d), dur_s=d.t.iloc[-1] - d.t.iloc[0],
                fs_med=1000 / np.median(dt), dt_max_ms=dt.max())


def first_disturbance(D, fs, thr):
    t = np.arange(0, D.t.iloc[-1], 1 / fs)
    m = np.sqrt(sum(np.interp(t, D.t, D[a]) ** 2 for a in "xyz"))
    rest = np.median(m[: int(fs)])
    dev = np.abs(lowpass(m, 20, fs) - rest) > thr
    if not dev.any():
        return np.nan, np.nan
    i0 = int(np.argmax(dev))
    quiet = np.abs(lowpass(m, 5, fs) - rest) < 0.5
    j, n = i0, int(0.4 * fs)
    while j < len(t) - n and not quiet[j:j + n].all():
        j += 1
    return t[i0], t[j]


def find_jump_and_lag(L, G, fs, sync_thr, max_lag=0.5):
    tG0, tG1 = first_disturbance(G, fs, sync_thr)
    tL0, _ = first_disturbance(L, fs, sync_thr)
    if np.isnan(tG0):
        return np.nan, 0.0, np.nan
    coarse = 0.0 if np.isnan(tL0) else tG0 - tL0
    if abs(coarse) <= max_lag:
        coarse = 0.0
    t = np.arange(0, G.t.iloc[-1], 1 / fs)
    mG = np.sqrt(sum(np.interp(t, G.t, G[a]) ** 2 for a in "xyz"))
    mL = np.sqrt(sum(np.interp(t, L.t + coarse, L[a]) ** 2 for a in "xyz"))
    w = (t > tG0 - 0.5) & (t < tG1 + 0.3)
    a_, b_ = mL[w] - mL[w].mean(), mG[w] - mG[w].mean()
    c = np.correlate(b_, a_, "full")
    lags = np.arange(-len(a_) + 1, len(a_)) / fs
    k = np.abs(lags) <= max_lag
    return tG0, coarse + lags[k][np.argmax(c[k])], tG1


def build_signals(L, G, p):
    fs = p["fs"]
    t = np.arange(0, min(L.t.iloc[-1], G.t.iloc[-1] - p["lag"]), 1 / fs)
    sig = pd.DataFrame({"t": t})
    sig["ml"] = lowpass(np.interp(t, L.t, L[p["ml_axis"]]) * p["ml_sign"], p["fc_l5"], fs)
    sig["ap"] = lowpass(np.interp(t, L.t, L[p["ap_axis"]]) * p["ap_sign"], p["fc_l5"], fs)
    sig["gy"] = lowpass(np.interp(t, G.t - p["lag"], G[p["leg_axis"]]), p["fc_leg"], fs)
    for a in "xyz":
        sig[f"leg_{a}"] = lowpass(np.interp(t, G.t - p["lag"], G[a]), 3.0, fs)
    return sig


def detect_bursts(sig, p, t_start):
    fs, gy, t = p["fs"], sig.gy.values, sig.t.values
    env = lowpass(np.abs(gy - np.median(gy)), 2.0, fs)
    act = env > p["burst_thr"]
    segs, s = [], None
    for i, v in enumerate(act):
        if v and s is None:
            s = i
        if not v and s is not None:
            segs.append([s, i]); s = None
    if s is not None:
        segs.append([s, len(act) - 1])
    merged = []
    for a, b in segs:
        if merged and (a - merged[-1][1]) < p["burst_merge"] * fs:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(t[a], t[b]) for a, b in merged
            if t[a] > t_start and (t[b] - t[a]) >= p["burst_min_dur"]]


def quiet_baseline(sig, ho, fs, length=0.5, earliest=2.6, latest=0.6):
    t = sig.t.values
    best = None
    for s0 in np.arange(ho - earliest, ho - latest - length + 1e-9, 0.05):
        w = (t >= s0) & (t < s0 + length)
        if w.sum() < 5:
            continue
        sc = sig.ml.values[w].std() + sig.ap.values[w].std() + sig.gy.values[w].std()
        if best is None or sc < best[0]:
            best = (sc, w)
    return best[1] if best else (t >= ho - 1.5) & (t < ho - 1.0)


def leg_baseline(sig, bs, length=0.5):
    t = sig.t.values
    V = sig[["leg_x", "leg_y", "leg_z"]].values
    best = None
    for s0 in np.arange(bs - 2.5, bs - 0.3 - length + 1e-9, 0.05):
        w = (t >= s0) & (t < s0 + length)
        if w.sum() < 5:
            continue
        sc = sig.gy.values[w].std() + V[w].std(0).sum()
        if best is None or sc < best[0]:
            best = (sc, w)
    return best[1] if best else (t >= bs - 1.5) & (t < bs - 1.0)


def detect_heel_off(sig, bs, p):
    """Heel-off = first peak of the leg vertical signal exceeding the leg resting level by ho_thr."""
    t, gy = sig.t.values, sig.gy.values
    bw = leg_baseline(sig, bs)
    g0 = gy[bw].mean()
    seg = np.flatnonzero((t > bs - 1.0) & (t < bs + 1.5))
    above = gy[seg] - g0 > p["ho_thr"]
    if not above.any():
        return np.nan
    pk = seg[np.argmax(above)]
    while pk + 1 < len(gy) and gy[pk + 1] > gy[pk]:
        pk += 1
    return t[pk]


def apa_metrics(sig, ho, p):
    """APA metrics (forward step) for ML and AP between baseline end and heel-off.
    Onset: walk back from the APA peak to the last sample inside baseline ± max(k·SD, minimum)."""
    fs, t = p["fs"], sig.t.values
    bw = quiet_baseline(sig, ho, fs, p["base_len"])
    bend = np.flatnonzero(bw)[-1]
    jho = int(np.argmin(np.abs(t - ho)))
    win = np.arange(bend, jho + 1)
    min_abs = p["min_abs_grav"] if p["grav"] else p["min_abs_lin"]
    out = {"baseline_start_s": round(t[bw][0], 2), "baseline_end_s": round(t[bw][-1], 2)}
    for name in ("ML", "AP"):
        s = sig[name.lower()].values
        b0, sd = s[bw].mean(), max(s[bw].std(), 0.01)
        thr = max(p["k_sd"] * sd, min_abs)
        if name == "AP":
            d = 1                                   # forward steps only
        else:                                       # dominant ML direction in [ho-0.8, ho-0.2]
            ww = (t > ho - 0.8) & (t < ho - 0.2)
            dv = s[ww] - b0
            d = 1 if dv[np.argmax(np.abs(dv))] > 0 else -1
        dev = d * (s[win] - b0)
        ipk = int(np.argmax(dev))
        onset = None
        if dev[ipk] > thr:
            n = ipk
            while n > 0 and dev[n - 1] > thr:
                n -= 1
            onset = n
        out[f"{name}_direction"] = "+" if d > 0 else "−"
        out[f"{name}_threshold_m_s2"] = round(thr, 3)
        if onset is None:
            for k in ("onset_rel_HO_ms", "onset_s", "peak_m_s2", "peak_rel_HO_ms", "dv_m_s"):
                out[f"{name}_{k}"] = np.nan
            continue
        pw = win[onset:]
        pki = pw[np.argmax(d * (s[pw] - b0))]
        out[f"{name}_onset_rel_HO_ms"] = round((t[win[onset]] - ho) * 1000)
        out[f"{name}_onset_s"] = round(t[win[onset]], 3)
        out[f"{name}_peak_m_s2"] = round(s[pki] - b0, 3)
        out[f"{name}_peak_rel_HO_ms"] = round((t[pki] - ho) * 1000)
        out[f"{name}_dv_m_s"] = round(np.trapezoid(s[pw] - b0, dx=1 / fs), 3)
    mlr = sig.ml.values - sig.ml.values[bw].mean()
    apr = sig.ap.values - sig.ap.values[bw].mean()
    res_h = np.hypot(mlr, apr)
    ipk = win[np.argmax(res_h[win])]
    out["RES_peak_m_s2"] = round(res_h[ipk], 3)
    out["RES_peak_rel_HO_ms"] = round((t[ipk] - ho) * 1000)
    out["RES_angle_deg"] = round(np.degrees(np.arctan2(mlr[ipk], apr[ipk])), 1)
    return out, bw


def run_pipeline(L, G, p):
    """Sync, filter and detect steps. Every step after the sync event is labelled forward."""
    tj, lag_auto, tj_end = find_jump_and_lag(L, G, p["fs"], p["sync_thr"])
    if np.isnan(tj):
        raise ValueError("Sync event not found on the leg sensor (lower the sync threshold).")
    if p.get("lag") is None:
        p["lag"] = lag_auto
    sig = build_signals(L, G, p)
    bursts = detect_bursts(sig, p, tj_end - p["lag"] + p["post_jump"])
    ev = []
    for i, (bs, be) in enumerate(bursts, 1):
        ho = detect_heel_off(sig, bs, p)
        ev.append(dict(event=i, burst_start_s=round(bs, 2), burst_end_s=round(be, 2), type=FWD,
                       HO_auto_s=round(ho, 3) if not np.isnan(ho) else np.nan))
    ev = pd.DataFrame(ev, columns=["event", "burst_start_s", "burst_end_s", "type", "HO_auto_s"])
    return dict(t_jump=tj, t_jump_end=tj_end, lag_auto=lag_auto, sig=sig, events=ev)


def compute_results(sig, ev, p, gL, excl_events=()):
    """APA metrics for every forward step with a heel-off. Returns per-step table and baselines."""
    sel = ev[(ev.type == FWD) & ev.HO_s.notna()].reset_index(drop=True)
    rows, bws = [], {}
    for k, r in sel.iterrows():
        m, bw = apa_metrics(sig, r.HO_s, p)
        rows.append(dict(step=k + 1, event=int(r.event), L5_gravity=gL, HO_s=r.HO_s,
                         HO_source="manual" if pd.notna(r.get("HO_manual_s", np.nan)) else "auto", **m))
        bws[k + 1] = bw
    res = pd.DataFrame(rows)
    if res.empty:
        return res, bws
    on_ = res[["ML_onset_rel_HO_ms", "AP_onset_rel_HO_ms"]].min(axis=1)
    res["APA_onset_ms"] = on_
    res["APA_duration_ms"] = -on_
    if p["st_sign"] < 0:
        res["RES_angle_deg"] = -res["RES_angle_deg"]
    res.insert(0, "included", ~res.event.isin(list(excl_events)))
    return res, bws


def summary_row(subj, res_all, p, gL, lag):
    inc = res_all[res_all.included] if not res_all.empty else res_all
    row = {"subject": subj, "step_type": FWD, "L5_gravity": gL, "lag_s": round(lag, 3),
           "n_steps_detected": len(res_all), "n_steps": len(inc),
           "n_steps_with_APA_onset": int(inc["APA_onset_ms"].notna().sum()) if len(inc) else 0,
           "excluded_events": ", ".join(map(str, res_all.loc[~res_all.included, "event"]))
           if len(res_all) else "-",
           "k_SD": p["k_sd"]}
    row["excluded_events"] = row["excluded_events"] or "-"
    for c in APA_VARS:
        row[f"{c}_mean"] = round(inc[c].mean(), 3) if len(inc) else np.nan
        row[f"{c}_sd"] = round(inc[c].std(), 3) if len(inc) else np.nan
    return row


def sheet_name(s, used):
    base = re.sub(r"[\[\]:*?/\\]", "_", str(s)).strip()[:31] or "subject"
    name, k = base, 2
    while name.lower() in {u.lower() for u in used}:
        suf = f"_{k}"
        name = base[:31 - len(suf)] + suf
        k += 1
    used.add(name)
    return name


def build_workbook(subjects, params):
    """subjects: list of dicts with keys id, res_all, row. Returns xlsx bytes."""
    from openpyxl.styles import Alignment, Font

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame([s["row"] for s in subjects]).to_excel(xw, sheet_name="Summary", index=False)
        used = {"Summary", "Parameters", "Variables"}
        cols = ["included", "step", "event", "HO_s", "HO_source"] + list(APA_VARS)
        stat_rows = {}
        for s in subjects:
            name = sheet_name(s["id"], used)
            ra = s["res_all"]
            if ra.empty:
                pd.DataFrame({"note": ["No forward step with a heel-off was detected."]}).to_excel(
                    xw, sheet_name=name, index=False)
                continue
            per = ra[cols].rename(columns=APA_VARS)
            per.to_excel(xw, sheet_name=name, index=False)
            inc = ra[ra.included]
            stats = pd.DataFrame({"included": ["Mean (included)", "SD (included)", "n"]})
            for c in ["step", "event", "HO_s", "HO_source"]:
                stats[c] = ""
            for c in APA_VARS:
                stats[APA_VARS[c]] = [round(inc[c].mean(), 3), round(inc[c].std(), 3),
                                      int(inc[c].notna().sum())]
            start = len(per) + 2
            stats.to_excel(xw, sheet_name=name, index=False, header=False, startrow=start + 1)
            stat_rows[name] = start + 2
        pd.DataFrame(list(params.items()), columns=["parameter", "value"]).to_excel(
            xw, sheet_name="Parameters", index=False)
        pd.DataFrame({"column": list(APA_VARS), "label": list(APA_VARS.values()),
                      "meaning": [VAR_HELP[c] for c in APA_VARS]}).to_excel(
            xw, sheet_name="Variables", index=False)

        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            ws.row_dimensions[1].height = 45
            for col in ws.columns:
                lens = [len(str(c.value)) for c in col[1:] if c.value is not None]
                ws.column_dimensions[col[0].column_letter].width = min(max([12] + lens) + 2, 60)
            if ws.title in stat_rows:
                for r in range(stat_rows[ws.title], stat_rows[ws.title] + 3):
                    for cell in ws[r]:
                        cell.font = Font(bold=True)
    return buf.getvalue()


# =============================================================================
# Figures
# =============================================================================

def fig_sync(L, G, R, p):
    t = R["sig"].t.values
    w = (t > R["t_jump"] - p["lag"] - 1.0) & (t < R["t_jump_end"] - p["lag"] + 1.0)
    mL = np.sqrt(sum(np.interp(t, L.t, L[a]) ** 2 for a in "xyz"))
    mG = np.sqrt(sum(np.interp(t, G.t - p["lag"], G[a]) ** 2 for a in "xyz"))
    fig = go.Figure()
    fig.add_scatter(x=t[w], y=mL[w], name="|a| L5")
    fig.add_scatter(x=t[w], y=mG[w], name="|a| leg (synchronised)")
    fig.update_layout(height=300, xaxis_title="Time (s)", yaxis_title="m/s²", margin=dict(t=20, b=40))
    return fig


def fig_overview(sig, ev, res, R, p):
    t = sig.t.values
    side = "stance" if p["st_sign"] > 0 else "swing"
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=("Leg vertical", f"L5 ML (+ = {side} side)", "L5 AP (+ forward)"))
    fig.add_scatter(x=t, y=sig.gy, line=dict(color="black", width=1), row=1, col=1)
    fig.add_scatter(x=t, y=sig.ml, line=dict(color="#1f77b4", width=1), row=2, col=1)
    fig.add_scatter(x=t, y=sig.ap, line=dict(color="#ff7f0e", width=1), row=3, col=1)
    excl = set(res.loc[~res.included, "event"]) if not res.empty else set()
    for _, r in ev.iterrows():
        col = ("rgba(128,128,128,0.15)" if (r.type == IGN or r.event in excl)
               else "rgba(46,160,67,0.15)")
        fig.add_vrect(x0=r.burst_start_s, x1=r.burst_end_s, fillcolor=col, line_width=0, row="all", col=1)
        fig.add_annotation(x=r.burst_start_s, y=1, yref="y domain", text=str(r.event), showarrow=False,
                           xanchor="left", font=dict(size=10), row=1, col=1)
    fig.add_vrect(x0=R["t_jump"] - p["lag"], x1=R["t_jump_end"] - p["lag"],
                  fillcolor="rgba(148,103,189,0.18)", line_width=0, row="all", col=1)
    for _, r in ev[(ev.type == FWD) & ev.HO_s.notna()].iterrows():
        fig.add_vline(x=r.HO_s, line=dict(color="red", dash="dash", width=1))
    for _, r in res.iterrows():
        if pd.notna(r.get("ML_onset_s")):
            fig.add_vline(x=r.ML_onset_s, line=dict(color="#1f77b4", dash="dot", width=1), row=2, col=1)
        if pd.notna(r.get("AP_onset_s")):
            fig.add_vline(x=r.AP_onset_s, line=dict(color="#ff7f0e", dash="dot", width=1), row=3, col=1)
    fig.update_layout(height=600, showlegend=False, margin=dict(t=40, b=40))
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    return fig


def fig_aligned(sig, res, bws, p):
    t, fs = sig.t.values, p["fs"]
    side = "stance" if p["st_sign"] > 0 else "swing"
    tt = np.arange(-1.0, 1.0 + 1e-9, 1 / fs)
    colors = pc.qualitative.Plotly + pc.qualitative.D3
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Leg vertical (re. baseline)", f"L5 ML (+ = {side} side, re. baseline)",
                                        "L5 AP (re. baseline, + forward)"))
    mean_on = {nm: np.nanmean(res[f"{nm}_onset_rel_HO_ms"]) / 1000
               if res[f"{nm}_onset_rel_HO_ms"].notna().any() else np.nan for nm in ("ML", "AP")}
    sd_on = {nm: np.nanstd(res[f"{nm}_onset_rel_HO_ms"]) / 1000
             if res[f"{nm}_onset_rel_HO_ms"].notna().any() else np.nan for nm in ("ML", "AP")}
    apa_start = {1: np.nanmin(list(mean_on.values())) if any(~np.isnan(list(mean_on.values()))) else np.nan,
                 2: mean_on["ML"], 3: mean_on["AP"]}
    for row in (1, 2, 3):
        if not np.isnan(apa_start[row]):
            fig.add_vrect(x0=apa_start[row], x1=0, fillcolor="rgba(255,215,0,0.20)", line_width=0,
                          row=row, col=1)
    for row, nm in ((2, "ML"), (3, "AP")):
        if not np.isnan(mean_on[nm]):
            fig.add_annotation(x=mean_on[nm] / 2, y=1, yref=f"y{row} domain", showarrow=False,
                               text=f"onset {mean_on[nm]*1000:.0f} ± {sd_on[nm]*1000:.0f} ms",
                               font=dict(size=11), row=row, col=1)
    for row, col in ((1, "gy"), (2, "ml"), (3, "ap")):
        s = sig[col].values
        M = []
        for j, (_, rr) in enumerate(res.iterrows()):
            cor = colors[j % len(colors)]
            y = np.interp(tt + rr.HO_s, t, s) - s[bws[rr.step]].mean()
            M.append(y)
            fig.add_scatter(x=tt, y=y, line=dict(color=cor, width=1.2), opacity=0.75,
                            name=f"Step {rr.step} (ev {rr.event})", legendgroup=f"s{rr.step}",
                            showlegend=(row == 1), row=row, col=1)
            if col in ("ml", "ap"):
                nm = col.upper()
                if pd.notna(rr[f"{nm}_onset_rel_HO_ms"]):
                    xo = rr[f"{nm}_onset_rel_HO_ms"] / 1000
                    fig.add_scatter(x=[xo], y=[np.interp(xo, tt, y)], mode="markers",
                                    marker=dict(color=cor, size=8, symbol="circle-open", line=dict(width=2)),
                                    legendgroup=f"s{rr.step}", showlegend=False, row=row, col=1)
                    fig.add_scatter(x=[rr[f"{nm}_peak_rel_HO_ms"] / 1000], y=[rr[f"{nm}_peak_m_s2"]],
                                    mode="markers", marker=dict(color=cor, size=9, symbol="x"),
                                    legendgroup=f"s{rr.step}", showlegend=False, row=row, col=1)
        M = np.array(M)
        mu, sd = M.mean(0), M.std(0)
        fig.add_scatter(x=np.r_[tt, tt[::-1]], y=np.r_[mu + sd, (mu - sd)[::-1]], fill="toself",
                        fillcolor="rgba(0,0,0,0.08)", line=dict(width=0), hoverinfo="skip",
                        name="Mean ± SD", legendgroup="mean", showlegend=(row == 1), row=row, col=1)
        fig.add_scatter(x=tt, y=mu, line=dict(color="black", width=3), name=f"Mean (n={len(M)})",
                        legendgroup="mean", showlegend=(row == 1), row=row, col=1)
    fig.add_vline(x=0, line=dict(color="red", dash="dash"))
    fig.update_yaxes(title_text="m/s²")
    fig.update_xaxes(title_text="Time relative to heel-off (s)", row=3, col=1)
    fig.update_xaxes(range=[-1, 1], dtick=0.25)
    fig.update_layout(height=750, margin=dict(t=40, b=40, l=60, r=20),
                      legend=dict(groupclick="togglegroup"))
    return fig


# =============================================================================
# Streamlit interface
# =============================================================================

@st.cache_data(show_spinner=False)
def process_files(l5_bytes, leg_bytes, p_items):
    p = dict(p_items)
    L, G = read_acc(io.BytesIO(l5_bytes)), read_acc(io.BytesIO(leg_bytes))
    R = run_pipeline(L, G, p)
    return L, G, R, p


def stretch(fn, *a, **k):
    try:
        return fn(*a, width="stretch", **k)
    except TypeError:
        return fn(*a, use_container_width=True, **k)


def main():
    st.set_page_config(page_title=f"Forward-step APA (batch) — {AUTHOR}", layout="wide",
                       menu_items={"About": f"**Step APA analysis — batch** · v{APP_VERSION}\n\n"
                                            f"Developed by **{AUTHOR}**"
                                            + (f" — {AUTHOR_INFO}" if AUTHOR_INFO else "")})
    st.title("Forward-step APA — batch analysis (L5 × leg)")
    st.caption(f"Developed by **{AUTHOR}**" + (f" · {AUTHOR_INFO}" if AUTHOR_INFO else "")
               + f" · v{APP_VERSION} · every step after the sync event is analysed as a forward step.")

    # ---------------- Sidebar: settings (same for all subjects) ----------------
    sb = st.sidebar
    sb.header("Subjects")
    n_subj = int(sb.number_input("Number of subjects", 1, 20, 5, 1))
    sb.header("Analysis")
    k_sd = sb.slider("APA threshold = k × L5 baseline SD", 1.0, 10.0, 5.0, 0.5,
                     help="APA onset = last moment the signal was inside baseline ± max(k·SD, minimum) "
                          "before the APA peak. Report the value you use.")
    sb.markdown("---")
    sb.caption("Advanced settings — the defaults fit the current protocol.")
    with sb.expander("Sensor axes"):
        c1, c2 = st.columns(2)
        ml_axis = c1.selectbox("L5 ML axis", ["x", "y", "z"], 0)
        ml_sign = c2.selectbox("ML sign", [1, -1], 0)
        ap_axis = c1.selectbox("L5 AP axis", ["x", "y", "z"], 2)
        ap_sign = c2.selectbox("AP sign (+ = forward)", [1, -1], 1)
        leg_axis = c1.selectbox("Leg vertical axis", ["x", "y", "z"], 1)
        stance_side = st.radio("Stance foot is on the ML …", ["+ side", "− side"], 0, horizontal=True)
        st_sign = 1 if stance_side.startswith("+") else -1
    with sb.expander("Filters and synchronisation"):
        fs = int(st.number_input("Resampling rate (Hz)", 50, 500, 100, 10))
        fc_l5 = st.slider("L5 low-pass (Hz)", 1.0, 20.0, 5.0, 0.5)
        fc_leg = st.slider("Leg low-pass (Hz)", 2.0, 30.0, 10.0, 0.5)
        sync_thr = st.slider("Sync event threshold (m/s²)", 0.5, 5.0, 1.5, 0.1)
        st.caption("Manual lag can be set per subject, in each subject's tab.")
    with sb.expander("Step detection"):
        burst_thr = st.slider("Leg activity threshold (m/s²)", 0.05, 1.5, 0.25, 0.05)
        burst_merge = st.slider("Merge bursts closer than (s)", 0.1, 2.0, 0.8, 0.1)
        burst_min_dur = st.slider("Minimum burst duration (s)", 0.1, 1.5, 0.3, 0.05)
        post_jump = st.slider("Ignore X s after the sync event", 0.0, 3.0, 0.5, 0.1)
    with sb.expander("Heel-off and APA thresholds"):
        ho_thr = st.slider("Heel-off: minimum leg peak above baseline (m/s²)", 0.3, 3.0, 0.8, 0.1)
        min_abs_grav = st.slider("APA minimum band width — data with gravity (m/s²)", 0.0, 0.5, 0.10, 0.01)
        min_abs_lin = st.slider("APA minimum band width — linear acceleration (m/s²)", 0.0, 0.5, 0.04, 0.01)
        base_len = st.slider("L5 baseline window (s)", 0.2, 1.0, 0.5, 0.05)
    sb.markdown("---")
    sb.caption(f"© {AUTHOR} · v{APP_VERSION}")

    p_base = dict(fs=fs, fc_l5=fc_l5, fc_leg=fc_leg, ml_axis=ml_axis, ml_sign=ml_sign,
                  ap_axis=ap_axis, ap_sign=ap_sign, leg_axis=leg_axis, sync_thr=sync_thr,
                  burst_thr=burst_thr, burst_merge=burst_merge, burst_min_dur=burst_min_dur,
                  post_jump=post_jump, ho_thr=ho_thr)
    p_apa = dict(k_sd=k_sd, min_abs_grav=min_abs_grav, min_abs_lin=min_abs_lin,
                 base_len=base_len, st_sign=st_sign)

    # ---------------- 1. Files ----------------
    st.subheader("1. Files (one L5 + one leg file per subject)")
    slots = []
    for i in range(n_subj):
        c = st.columns([1.2, 2, 2])
        fl5 = c[1].file_uploader(f"L5 — subject {i+1}", type=["txt", "csv"], key=f"l5_{i}")
        flg = c[2].file_uploader(f"Leg — subject {i+1}", type=["txt", "csv"], key=f"leg_{i}")
        default = (fl5.name.replace("\\", "/").split("/")[-1].rsplit(".", 1)[0]
                   if fl5 else f"S{i+1:02d}")
        sid = c[0].text_input(f"Subject {i+1} ID", default, key=f"id_{i}_{default}")
        slots.append((i, sid.strip() or f"S{i+1:02d}", fl5, flg))

    ready = [s for s in slots if s[2] and s[3]]
    incomplete = [s[1] for s in slots if bool(s[2]) != bool(s[3])]
    if incomplete:
        st.warning(f"Missing one of the two files for: {', '.join(incomplete)}.")
    if not ready:
        st.info("Upload the L5 and leg files of at least one subject to start.")
        st.stop()
    ids = [s[1] for s in ready]
    if len(set(ids)) < len(ids):
        st.warning("Some subject IDs are repeated — the Excel sheets will get a suffix (_2, _3 …).")

    # ---------------- 2. Per-subject analysis ----------------
    st.subheader("2. Analysis")
    tabs = st.tabs(["Summary"] + [f"{s[1]}" for s in ready])
    subjects = []
    for tab, (i, sid, fl5, flg) in zip(tabs[1:], ready):
        with tab:
            c = st.columns([1, 1, 3])
            lag_manual = c[0].checkbox("Set lag manually", key=f"lagm_{i}")
            lag_val = c[1].number_input("Leg lag re. L5 (s)", -2.0, 2.0, 0.0, 0.01, key=f"lagv_{i}",
                                        disabled=not lag_manual)
            p_proc = dict(p_base, lag=lag_val if lag_manual else None)
            try:
                with st.spinner(f"Processing {sid}…"):
                    L, G, R, p = process_files(fl5.getvalue(), flg.getvalue(), tuple(sorted(p_proc.items())))
            except Exception as e:
                st.error(f"{sid}: could not be processed — {e}")
                continue
            gL = has_gravity(L)
            p.update(p_apa, grav=gL)
            sig, t = R["sig"], R["sig"].t.values

            iL, iG = sampling_info(L), sampling_info(G)
            m = st.columns(4)
            m[0].metric("L5: samples / rate", f"{iL['n']} / {iL['fs_med']:.0f} Hz")
            m[1].metric("Leg: samples / rate", f"{iG['n']} / {iG['fs_med']:.0f} Hz")
            m[2].metric("Sync event (leg)", f"{R['t_jump']:.2f}–{R['t_jump_end']:.2f} s")
            m[3].metric("Lag used", f"{p['lag']:.3f} s", f"auto = {R['lag_auto']:.3f} s", delta_color="off")
            st.caption(f"L5 data: **{'with gravity' if gL else 'linear acceleration (gravity removed)'}**.")

            with st.expander("Check the alignment at the sync event"):
                stretch(st.plotly_chart, fig_sync(L, G, R, p), key=f"sync_{i}")

            ev = R["events"].copy()
            if ev.empty:
                st.warning("No steps detected. Lower the leg activity threshold in the sidebar.")
                res_all, bws = pd.DataFrame(), {}
            else:
                ev["HO_manual_s"] = np.nan
                with st.expander("Event table (set type = ignore, or a manual heel-off)"):
                    ev = stretch(st.data_editor, ev, hide_index=True, key=f"ev_{i}",
                                 column_config={
                                     "type": st.column_config.SelectboxColumn(options=[FWD, IGN]),
                                     "HO_manual_s": st.column_config.NumberColumn(format="%.3f")},
                                 disabled=["event", "burst_start_s", "burst_end_s", "HO_auto_s"])
                ev["HO_s"] = ev.HO_manual_s.fillna(ev.HO_auto_s)
                cand = ev[(ev.type == FWD) & ev.HO_s.notna()].event.astype(int).tolist()
                excl = st.multiselect("Exclude events from the means (kept in the sheet as included = False)",
                                      cand, key=f"excl_{i}", format_func=lambda e: f"Event {e}")
                res_all, bws = compute_results(sig, ev, p, gL, excl)

            if res_all.empty:
                st.warning("No forward step with a heel-off.")
            else:
                # sanity check: forward steps should push L5 forward before heel-off
                pos = []
                for _, r in res_all.iterrows():
                    w_ = (t > r.HO_s - 0.4) & (t <= r.HO_s)
                    pos.append(sig.ap.values[w_].mean() - sig.ap.values[bws[r.step]].mean() > 0)
                if len(pos) >= 3 and np.mean(pos) < 0.65:
                    st.warning(f"L5 AP is forward before heel-off in only {np.mean(pos):.0%} of the steps. "
                               "Check the AP axis/sign and whether some events are not real steps.")
                stretch(st.plotly_chart, fig_overview(sig, ev, res_all, R, p), key=f"ov_{i}")
                st.caption("Green = analysed step (number = event) · grey = ignored/excluded · purple = sync "
                           "event · red dashed = heel-off · dotted = APA onset (ML blue, AP orange).")
                res = res_all[res_all.included].reset_index(drop=True)
                if not res.empty:
                    stretch(st.plotly_chart, fig_aligned(sig, res, bws, p), key=f"al_{i}")
                st.markdown("**Per step**")
                stretch(st.dataframe, res_all[["included", "step", "event", "HO_s", "HO_source"]
                                              + list(APA_VARS)].rename(columns=APA_VARS).round(3),
                        hide_index=True)
            subjects.append(dict(id=sid, res_all=res_all, row=summary_row(sid, res_all, p, gL, p["lag"])))

    # ---------------- Summary + Excel ----------------
    with tabs[0]:
        if not subjects:
            st.warning("No subject could be analysed.")
            st.stop()
        summ = pd.DataFrame([s["row"] for s in subjects])
        st.markdown(f"**One row per subject (n = {len(subjects)})** — means and SDs use included steps only.")
        stretch(st.dataframe, summ, hide_index=True)
        if summ["L5_gravity"].nunique() > 1:
            st.warning("Some L5 recordings include gravity and others do not — their amplitudes are "
                       "not comparable (see column L5_gravity).")
        params = dict(p_base, **p_apa, step_type="forward only", ho_method="first leg vertical peak",
                      onset_method="walk back from peak", app_version=APP_VERSION,
                      analysis_date=str(date.today()))
        params.pop("lag", None)
        xlsx = build_workbook(subjects, params)
        st.download_button("Download Excel (Summary + one sheet per subject)", xlsx,
                           f"APA_forward_batch_{date.today():%Y%m%d}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           type="primary")
        st.caption("Review each subject's tab (events, exclusions, manual heel-offs) before downloading — "
                   "the Excel reflects the current choices in every tab.")

    st.markdown("---")
    st.caption(f"Step APA analysis (batch) · developed by **{AUTHOR}**"
               + (f" · {AUTHOR_INFO}" if AUTHOR_INFO else "") + f" · v{APP_VERSION}")


if __name__ == "__main__":
    main()
