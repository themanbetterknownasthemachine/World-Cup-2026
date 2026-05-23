"""wm_model.py — Kernlogik des WM-2026-Prognosemodells.

Buendelt Elo-Wertung, Tor-Modell, Gruppen-Rekonstruktion und Turnier-Simulation,
damit Notebooks und das Automatik-Skript dieselbe, getestete Logik nutzen.

Zentrale Idee fuer die Live-Schicht: `simulate_tournament` akzeptiert ein Dict
`known` mit bereits gespielten Resultaten. Diese Spiele werden fixiert, alle
uebrigen ausgewuerfelt — so schaerft sich die Prognose mit jedem Spieltag.
"""
import numpy as np
import pandas as pd
import networkx as nx
import string
import json
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import poisson

DATA_URL = ("https://raw.githubusercontent.com/martj42/"
            "international_results/master/results.csv")
WM_START = pd.Timestamp("2026-06-11")
BASE, HFA = 1500.0, 100.0
MG = 11                       # max. Tore pro Team in der Score-Matrix


# ---------------------------------------------------------------- Daten
def load_data(local="data/results.csv"):
    """Laedt den Datensatz: zuerst lokal, sonst vom GitHub-Mirror (mit Retries
    und Caching). Robuster Loader analog zu den Notebooks.
    """
    local_path = Path(local)
    if local_path.exists():
        df_all = pd.read_csv(local_path, parse_dates=["date"])
    else:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df_all = None
        for attempt in range(1, 4):
            try:
                df_all = pd.read_csv(DATA_URL, parse_dates=["date"])
                df_all.to_csv(local_path, index=False)        # fuer naechstes Mal cachen
                break
            except Exception as e:
                print(f"[load_data] Versuch {attempt}/3 fehlgeschlagen: {e}")
        if df_all is None:
            raise RuntimeError("Konnte results.csv weder lokal noch via URL laden")
    df = df_all.dropna(subset=["home_score", "away_score"]).copy()
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    return df_all, df


# ---------------------------------------------------------------- Elo
def _k_factor(t):
    if t == "FIFA World Cup": return 60
    if "qualification" in t:  return 40
    if t in ("UEFA Euro", "Copa América", "African Cup of Nations",
             "AFC Asian Cup", "Gold Cup", "Confederations Cup"): return 50
    if t == "Friendly":       return 20
    return 30

def _g_mult(margin):
    m = abs(margin)
    if m <= 1: return 1.0
    if m == 2: return 1.5
    return (11 + m) / 8

def compute_elo(df, extra=None):
    """Elo chronologisch berechnen. `extra` = optionale Liste zusaetzlicher
    Spiele (dicts mit home_team, away_team, home_score, away_score, neutral,
    tournament) — z.B. echte WM-Resultate, die nach dem Datensatz kamen.

    Live-Resultate, die nach (date, home_team, away_team) bereits im Datensatz
    vorhanden sind, werden uebersprungen — verhindert Doppel-Updates, falls der
    Upstream-Datensatz waehrend des Turniers nachgepflegt wurde.

    Gibt (elo_dict, df_mit_pre_ratings) zurueck.
    """
    rows = df
    if extra:
        extra_df = pd.DataFrame(extra)
        extra_df["date"] = pd.to_datetime(extra_df["date"])
        existing = set(zip(df["date"].dt.date.tolist(),
                           df["home_team"].tolist(),
                           df["away_team"].tolist()))
        keep = [(d.date(), h, a) not in existing for d, h, a in
                zip(extra_df["date"], extra_df["home_team"], extra_df["away_team"])]
        extra_df = extra_df[keep]
        if len(extra_df) > 0:
            rows = pd.concat([df, extra_df], ignore_index=True)
            rows = rows.sort_values("date").reset_index(drop=True)
    elo = {}
    pre_h = np.zeros(len(rows)); pre_a = np.zeros(len(rows))
    for i, r in enumerate(rows.itertuples()):
        Rh = elo.get(r.home_team, BASE); Ra = elo.get(r.away_team, BASE)
        pre_h[i], pre_a[i] = Rh, Ra
        adv = HFA * (0 if r.neutral else 1)
        Eh = 1 / (1 + 10 ** (-(Rh + adv - Ra) / 400))
        Wh = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        d = _k_factor(r.tournament) * _g_mult(r.home_score - r.away_score) * (Wh - Eh)
        elo[r.home_team] = Rh + d
        elo[r.away_team] = Ra - d
    rows = rows.copy(); rows["pre_h"], rows["pre_a"] = pre_h, pre_a
    return elo, rows


# ---------------------------------------------------------------- Tor-Modell
def fit_goals(df_pre, ref=WM_START, cache=None):
    """Poisson-Regression: log(Tore) = b0 + b1*EloDiff/100 + b2*Heim.
    Fit nur auf Spielen vor `ref`. Gibt Koeffizienten b zurueck.

    Optionaler `cache`-Pfad: speichert/laedt b zur Vermeidung von Re-Fits
    bei identischer Trainingsbasis (Daten vor `ref` aendern sich waehrend des
    Turniers nicht). Cache-Key = Anzahl Zeilen + Tor-Summen; bei Aenderung der
    Trainingsbasis wird automatisch neu gefittet.
    """
    tr = df_pre[df_pre["date"] < ref]
    key = f"{len(tr)}-{int(tr['home_score'].sum())}-{int(tr['away_score'].sum())}"

    if cache:
        cf = Path(cache)
        if cf.exists():
            try:
                d = json.loads(cf.read_text(encoding="utf-8"))
                if d.get("key") == key:
                    return np.array(d["b"])
            except Exception:
                pass   # bei Cache-Fehler einfach neu fitten

    diff = np.concatenate([(tr["pre_h"] - tr["pre_a"]) / 100,
                           (tr["pre_a"] - tr["pre_h"]) / 100])
    home = np.concatenate([(1 - tr["neutral"].astype(float)), np.zeros(len(tr))])
    y = np.concatenate([tr["home_score"].values, tr["away_score"].values]).astype(float)
    def nll(b):
        eta = b[0] + b[1]*diff + b[2]*home
        return -np.sum(y * eta - np.exp(eta))
    b = minimize(nll, [0.0, 0.3, 0.2], method="L-BFGS-B").x

    if cache:
        try:
            cf = Path(cache)
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(json.dumps({"key": key, "b": b.tolist()}), encoding="utf-8")
        except Exception:
            pass
    return b


# ---------------------------------------------------------------- Gruppen
def reconstruct_groups(df_all):
    """Die 12 echten WM-2026-Gruppen aus den Spielpaarungen ableiten."""
    gs = df_all[(df_all["tournament"] == "FIFA World Cup")
                & (df_all["date"].between("2026-06-11", "2026-06-27"))]
    G = nx.Graph()
    for _, r in gs.iterrows():
        G.add_edge(r["home_team"], r["away_team"])
    comps = list(nx.connected_components(G))
    order, seen = [], set()
    for _, r in gs.sort_values("date").iterrows():
        for t in (r["home_team"], r["away_team"]):
            if t not in seen: seen.add(t); order.append(t)
    comp_of = {t: i for i, c in enumerate(comps) for t in c}
    first = {}
    for pos, t in enumerate(order): first.setdefault(comp_of[t], pos)
    labels = sorted(first, key=lambda c: first[c])
    return {L: sorted(comps[c]) for L, c in zip(string.ascii_uppercase, labels)}


# ---------------------------------------------------------------- Engine
class Engine:
    """Vorberechnete Score-Verteilungen aller Paare fuer schnelles Sampling."""
    def __init__(self, teams, elo, b):
        self.teams = teams
        self.wi = {t: i for i, t in enumerate(teams)}
        M = len(teams)
        R = np.array([elo.get(t, BASE) for t in teams])
        lam = lambda ra, rb: np.exp(b[0] + b[1]*(ra-rb)/100)   # neutral
        g = np.arange(MG)
        self.CUM = np.zeros((M, M, MG*MG)); self.PADV = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                lh, la = lam(R[i], R[j]), lam(R[j], R[i])
                SM = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la)); SM /= SM.sum()
                self.CUM[i, j] = np.cumsum(SM.ravel())
                ph, pa = np.tril(SM, -1).sum(), np.triu(SM, 1).sum()
                self.PADV[i, j] = ph / (ph + pa)

    def play(self, i, j, rng):
        return divmod(np.searchsorted(self.CUM[i, j], rng.random()), MG)

    def winner(self, i, j, rng):
        h, a = self.play(i, j, rng)
        return i if h > a else (j if a > h else (i if rng.random() < self.PADV[i, j] else j))


def _seed_order(N):
    o = [1, 2]
    while len(o) < N:
        c = len(o)*2+1; o = [v for s in o for v in (s, c-s)]
    return o
_SO = _seed_order(32)


# ---------------------------------------------------------------- Simulation
def simulate_tournament(groups, engine, known=None, n_sims=10000, seed=42):
    """Monte-Carlo-Simulation. `known` = dict frozenset({teamA,teamB}) ->
    (ToreA, ToreB) fuer bereits gespielte Spiele (werden fixiert).

    Gibt DataFrame mit Titel-/Finale-/Halbfinal-Wahrscheinlichkeit zurueck.
    """
    known = known or {}
    wi = engine.wi
    teams = engine.teams
    M = len(teams)
    rng = np.random.default_rng(seed)
    champ = np.zeros(M); finalist = np.zeros(M); semi = np.zeros(M)
    for _ in range(n_sims):
        third, qual = [], []
        for L, gt in groups.items():
            gi = [wi[t] for t in gt]
            pts = {k:0 for k in gi}; gf = {k:0 for k in gi}; ga = {k:0 for k in gi}
            for a in range(4):
                for b2 in range(a+1, 4):
                    key = frozenset((teams[gi[a]], teams[gi[b2]]))
                    if key in known:
                        h = known[key][teams[gi[a]]]; aw = known[key][teams[gi[b2]]]
                    else:
                        h, aw = engine.play(gi[a], gi[b2], rng)
                    gf[gi[a]]+=h; ga[gi[a]]+=aw; gf[gi[b2]]+=aw; ga[gi[b2]]+=h
                    if h>aw: pts[gi[a]]+=3
                    elif aw>h: pts[gi[b2]]+=3
                    else: pts[gi[a]]+=1; pts[gi[b2]]+=1
            rank = sorted(gi, key=lambda k:(pts[k], gf[k]-ga[k], gf[k]), reverse=True)
            for fin in (0,1,2):
                tm = rank[fin]; rec = (fin, pts[tm], gf[tm]-ga[tm], gf[tm], tm)
                (qual if fin<2 else third).append(rec)
        best3 = sorted(third, key=lambda x:(x[1],x[2],x[3]), reverse=True)[:8]
        q = [r[4] for r in sorted(qual+best3, key=lambda x:(x[0],-x[1],-x[2],-x[3]))]
        seeds = {s: q[s-1] for s in range(1,33)}
        bracket = [seeds[_SO[k]] for k in range(32)]
        sf = fin = None
        while len(bracket) > 1:
            if len(bracket) == 4: sf = list(bracket)
            if len(bracket) == 2: fin = list(bracket)
            nxt = []
            for k in range(0, len(bracket), 2):
                i, j = bracket[k], bracket[k+1]
                key = frozenset((teams[i], teams[j]))
                if key in known:
                    h = known[key][teams[i]]; aw = known[key][teams[j]]
                    if h > aw:
                        w = i
                    elif aw > h:
                        w = j
                    else:
                        # Remis in K.O. = Verlaengerung/Elfmeter, deren Sieger
                        # sich aus dem 90-Min-Score nicht ableiten laesst.
                        # Simulieren statt willkuerlich zu entscheiden.
                        # (Spaetere Verfeinerung: strStatus der API auswerten.)
                        w = engine.winner(i, j, rng)
                else:
                    w = engine.winner(i, j, rng)
                nxt.append(w)
            bracket = nxt
        champ[bracket[0]] += 1
        for x in fin:  finalist[x] += 1
        for x in sf:   semi[x] += 1

    return (pd.DataFrame({"Team": teams, "Titel": champ/n_sims,
                          "Finale": finalist/n_sims, "Halbfinale": semi/n_sims})
            .sort_values("Titel", ascending=False).reset_index(drop=True))


def build_forecast(local="data/results.csv", live_results=None, n_sims=10000,
                   fit_cache=None):
    """Kompletter Durchlauf: Daten -> Elo (inkl. Live-Resultate) -> Engine ->
    Simulation. `live_results` = Liste von Spiel-Dicts (echte WM-Resultate).
    `fit_cache` = optionaler Pfad zum Cache der Tor-Regressions-Koeffizienten.
    Gibt (DataFrame, Anzahl beruecksichtigter Live-Spiele) zurueck.
    """
    df_all, df = load_data(local)
    elo, df_pre = compute_elo(df, extra=live_results)
    b = fit_goals(df_pre, cache=fit_cache)
    groups = reconstruct_groups(df_all)
    teams = [t for g in groups.values() for t in g]
    engine = Engine(teams, elo, b)

    known = {}
    for m in (live_results or []):
        key = frozenset((m["home_team"], m["away_team"]))
        if key in known:
            continue                                  # Duplikat -> erstes gewinnt
        known[key] = {m["home_team"]: m["home_score"],
                      m["away_team"]: m["away_score"]}

    result = simulate_tournament(groups, engine, known=known, n_sims=n_sims)
    return result, len(known)
