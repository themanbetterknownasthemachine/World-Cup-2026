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
}
def _norm(name):
    return NAME_MAP.get(name, name)

# WM-2026-Gastgeber. Spiele dieser Teams gelten als Heimspiele (neutral=False),
# was den Heim-Vorteil-Bonus (HFA) in der Elo-Aktualisierung aktiviert.
# Vereinfachung: Wir nehmen an, ein Gastgeber als strHomeTeam = Spiel im eigenen
# Land. Edge case (Host spielt in anderem Host-Land) wird ignoriert.
HOST_TEAMS = {"United States", "Canada", "Mexico"}


def fetch_live_results():
    """Gespielte WM-2026-Spiele von TheSportsDB holen. Bei jedem Problem -> []."""
    try:
        import requests
        base = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}"
        leagues = requests.get(f"{base}/all_leagues.php", timeout=20).json()["leagues"]
        wc = [l for l in leagues if l["strLeague"] == "FIFA World Cup"]
        if not wc:
            return []
        lid = wc[0]["idLeague"]
        events = requests.get(f"{base}/eventsseason.php",
                              params={"id": lid, "s": "2026"}, timeout=20).json().get("events") or []
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
    result, n = wm.build_forecast(local=str(DATA), live_results=live, n_sims=10000,
                                  fit_cache=str(OUTDIR / "goal_coefs.json"))

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


if __name__ == "__main__":
    main()
