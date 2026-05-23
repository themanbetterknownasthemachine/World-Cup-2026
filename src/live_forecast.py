"""live_forecast.py — headless Lauf fuer die taegliche Automatik (GitHub Actions).

Holt die bereits gespielten WM-2026-Resultate von TheSportsDB, aktualisiert das
Modell und schreibt die aktuelle Prognose nach output/. Faellt zurueck auf die
Vorturnier-Prognose, wenn (noch) keine Resultate verfuegbar sind.

Aufruf:  python src/live_forecast.py
Env:     SPORTSDB_KEY  (Default "123" = Free-Tier)
"""
import os
import json
import datetime as dt
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import wm_model as wm
import pandas as pd

API_KEY = os.environ.get("SPORTSDB_KEY", "123")      # "123" = kostenloser Test-Key
DATA = ROOT / "data" / "results.csv"
OUTDIR = ROOT / "output"

# TheSportsDB nutzt teils andere Teamnamen als der Trainingsdatensatz.
# Diese Tabelle muss beim ersten echten Lauf gegen die API-Namen geprueft werden!
NAME_MAP = {
    "USA": "United States",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Cabo Verde": "Cape Verde",
    "Curacao": "Curaçao",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}
def _norm(name):
    return NAME_MAP.get(name, name)

# WM-2026-Gastgeber. Spiele dieser Teams gelten als Heimspiele (neutral=False),
# was den Heim-Vorteil-Bonus (HFA) in der Elo-Aktualisierung aktiviert.
# Vereinfachung: Wir nehmen an, ein Gastgeber als strHomeTeam = Spiel im eigenen
# Land. Edge case (Host spielt in anderem Host-Land) wird ignoriert.
HOST_TEAMS = {"United States", "Canada", "Mexico"}

# TheSportsDB-League-ID der FIFA World Cup. Hartcodiert, weil all_leagues.php im
# Free-Tier nur die 10 europaeischen Top-Ligen liefert und die WM nicht enthaelt.
# Per lookupleague.php?id=4429 verifiziert.
WC_LEAGUE_ID = 4429


def fetch_live_results():
    """Gespielte WM-2026-Spiele von TheSportsDB holen. Bei jedem Problem -> []."""
    try:
        import requests
        base = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}"
        events = requests.get(f"{base}/eventsseason.php",
                              params={"id": WC_LEAGUE_ID, "s": "2026"},
                              timeout=20).json().get("events") or []
        out = []
        for e in events:
            hs, as_ = e.get("intHomeScore"), e.get("intAwayScore")
            if hs in (None, "") or as_ in (None, ""):
                continue                                  # noch nicht gespielt
            home_norm = _norm(e["strHomeTeam"])
            away_norm = _norm(e["strAwayTeam"])
            out.append(dict(date=e["dateEvent"],
                            home_team=home_norm,
                            away_team=away_norm,
                            home_score=int(hs), away_score=int(as_),
                            neutral=(home_norm not in HOST_TEAMS),
                            tournament="FIFA World Cup"))
        return out
    except Exception as ex:
        print(f"[WARN] Live-Abruf fehlgeschlagen ({ex}) -> Vorturnier-Prognose.")
        return []


def main():
    live = fetch_live_results()
    print(f"Beruecksichtigte Live-Spiele: {len(live)}")
    OUTDIR.mkdir(exist_ok=True)
    result, n, ctx = wm.build_forecast(local=str(DATA), live_results=live, n_sims=10000,
                                       fit_cache=str(OUTDIR / "goal_coefs.json"),
                                       return_context=True)

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # JSON (maschinenlesbar)
    payload = {"updated": stamp, "live_matches": n,
               "forecast": result.to_dict(orient="records")}
    (OUTDIR / "forecast.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown (lesbar, direkt im Repo anzeigbar)
    lines = [f"# WM 2026 — Live-Prognose", "",
             f"Stand: {stamp} · beruecksichtigte Spiele: {n}", "",
             "| # | Team | Titel | Finale | Halbfinale |",
             "|--:|------|------:|-------:|-----------:|"]
    for i, r in result.head(15).iterrows():
        lines.append(f"| {i+1} | {r.Team} | {r.Titel:.1%} | {r.Finale:.1%} | {r.Halbfinale:.1%} |")
    (OUTDIR / "forecast.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Geschrieben: {OUTDIR/'forecast.md'} und forecast.json")

    # History: pro Tag eine Zeile je Team -> spaeter als Zeitreihe auswertbar.
    # Mehrfache Laeufe am selben UTC-Tag ueberschreiben die vorherigen Eintraege.
    history_path = OUTDIR / "history.csv"
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    new_rows = (result
                .rename(columns={"Team": "team", "Titel": "titel",
                                 "Finale": "finale", "Halbfinale": "halbfinale"})
                .assign(date=today, live_matches=n)
                [["date", "team", "titel", "finale", "halbfinale", "live_matches"]])

    if history_path.exists():
        try:
            hist = pd.read_csv(history_path)
            hist = hist[hist["date"] != today]   # gleicher Tag -> neuer Lauf gewinnt
            out_hist = pd.concat([hist, new_rows], ignore_index=True)
        except Exception as e:
            print(f"[WARN] history.csv unlesbar ({e}) -> wird neu geschrieben.")
            out_hist = new_rows
    else:
        out_hist = new_rows
    out_hist.to_csv(history_path, index=False, encoding="utf-8")
    print(f"History: {history_path} ({len(out_hist)} Zeilen)")

    # Pro-Spiel-Prognose (Ebene A): 1/X/2 + wahrscheinlichstes Resultat fuer
    # alle noch nicht gespielten WM-2026-Spiele.
    matches = wm.predict_remaining_matches(ctx["df_all"], ctx["engine"], ctx["known"])
    (OUTDIR / "matches.json").write_text(
        json.dumps({"updated": stamp, "live_matches": n,
                    "matches": matches.to_dict(orient="records")},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [f"# WM 2026 — Pro-Spiel-Prognose", "",
             f"Stand: {stamp} · noch offene Spiele: {len(matches)}", "",
             "Heimspiele eines Gastgebers (Mexico/USA/Canada) sind mit ⌂ markiert.", "",
             "| Datum | Spiel | 1 | X | 2 | Wahrsch. Resultat | P |",
             "|---|---|---:|---:|---:|:--:|---:|"]
    for _, r in matches.iterrows():
        marker = " ⌂" if r["home_advantage"] else ""
        lines.append(f"| {r['date']} | {r['home_team']}{marker} vs {r['away_team']} | "
                     f"{r['p_home']:.0%} | {r['p_draw']:.0%} | {r['p_away']:.0%} | "
                     f"{r['most_likely']} | {r['p_most_likely']:.1%} |")
    (OUTDIR / "matches.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Spiele: {OUTDIR/'matches.md'} und matches.json ({len(matches)} Spiele)")


if __name__ == "__main__":
    main()
