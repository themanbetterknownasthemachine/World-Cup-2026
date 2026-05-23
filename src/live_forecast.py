"""live_forecast.py — headless Lauf fuer die taegliche Automatik (GitHub Actions).

Holt die bereits gespielten WM-2026-Resultate von TheSportsDB, aktualisiert das
Modell und schreibt die aktuelle Prognose nach output/. Faellt zurueck auf die
Vorturnier-Prognose, wenn (noch) keine Resultate verfuegbar sind.

Aufruf:  python src/live_forecast.py
Env:     SPORTSDB_KEY  (Patreon-Key; ohne Key wird ein Warn-Hinweis ausgegeben
         und der Lauf nutzt nur die Vorturnier-Prognose)
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

# Paid Patreon-Key noetig (lokal: $env:SPORTSDB_KEY=..., CI: Repo-Secret).
# Der frueher freie Test-Key "123" liefert seit TheSportsDB-Umstellung keine
# WM-Daten mehr -> bewusst keinen Default mehr; fehlt der Key, gibt fetch_live_results
# eine Warnung aus und kehrt mit [] zurueck (Vorturnier-Prognose bleibt erhalten).
API_KEY = os.environ.get("SPORTSDB_KEY", "")
DATA = ROOT / "data" / "results.csv"
OUTDIR = ROOT / "output"
ARCHIVE = OUTDIR / "predictions_archive"

# TheSportsDB nutzt teils andere Teamnamen als der Trainingsdatensatz.
# Verifiziert 2026-05-23 gegen den WM-2026-Spielplan der API: alle 48 Teams werden
# aufgeloest. Aktiv noetig sind nur "USA" und "Bosnia-Herzegovina" — die uebrigen
# vier Eintraege ("Korea Republic", "IR Iran", "Cabo Verde", "Curacao") greifen
# aktuell nicht (die API liefert die kanonischen Namen direkt), bleiben aber als
# Sicherheitsnetz drin, falls TheSportsDB die Schreibweise spaeter umstellt.
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
    if not API_KEY:
        print("[WARN] SPORTSDB_KEY nicht gesetzt -> Vorturnier-Prognose.")
        return []
    try:
        import requests
        masked = f"{API_KEY[:2]}..{API_KEY[-2:]}" if len(API_KEY) > 4 else "***"
        print(f"[INFO] TheSportsDB-Abruf mit Key {masked} (Liga {WC_LEAGUE_ID}, Saison 2026).")
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


def archive_pre_match_forecasts(matches_df, today):
    """Friere die Pre-Match-Wahrscheinlichkeiten ein, bevor das Spiel angepfiffen wird.
    Pro Spieltag eine JSON-Datei. Sobald ein Spiel gespielt ist, faellt es aus
    `matches_df` raus -> die letzte Schreibung vor Anpfiff bleibt automatisch stehen
    und dient als Referenz fuer den Prognose-vs-Realitaet-Vergleich."""
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    upcoming = matches_df[matches_df["date"] >= today]
    for date, grp in upcoming.groupby("date"):
        path = ARCHIVE / f"{date}.json"
        path.write_text(grp.to_json(orient="records"), encoding="utf-8")


def build_results_payload(live):
    """Paare jedes gespielte Spiel mit seiner letzten archivierten Pre-Match-Prognose."""
    rows = []
    for m in live:
        date = m["date"]
        pred = None
        arc_path = ARCHIVE / f"{date}.json"
        if arc_path.exists():
            try:
                for p in json.loads(arc_path.read_text(encoding="utf-8")):
                    if p["home_team"] == m["home_team"] and p["away_team"] == m["away_team"]:
                        pred = p
                        break
            except Exception as e:
                print(f"[WARN] Archiv {arc_path.name} unlesbar ({e}).")
        if m["home_score"] > m["away_score"]:
            winner = "home"
        elif m["away_score"] > m["home_score"]:
            winner = "away"
        else:
            winner = "draw"
        # War die Prognose richtig? (hoechste der 3 W-keiten == tatsaechlicher Ausgang)
        pred_correct = None
        upset = False
        if pred is not None:
            probs = {"home": pred["p_home"], "draw": pred["p_draw"], "away": pred["p_away"]}
            pred_winner = max(probs, key=probs.get)
            pred_correct = (pred_winner == winner)
            # Upset: tatsaechlicher Sieger hatte vor dem Spiel <30% W-keit
            upset = (probs[winner] < 0.30) and (winner != "draw")
        rows.append({
            **m,
            "winner": winner,
            "score": f"{m['home_score']}:{m['away_score']}",
            "prediction": pred,
            "prediction_correct": pred_correct,
            "is_upset": upset,
        })
    return rows


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

    # Pre-Match-Snapshots einfrieren, damit der Dashboard-Vergleich Prognose vs.
    # Realitaet auch nach Anpfiff noch weiss, was wir vorher gesagt haben.
    archive_pre_match_forecasts(matches, today)

    # Gespielte Spiele + ihre archivierten Prognosen -> results.json fuer das Dashboard.
    results_payload = build_results_payload(live)
    correct = sum(1 for r in results_payload if r["prediction_correct"] is True)
    rated = sum(1 for r in results_payload if r["prediction_correct"] is not None)
    (OUTDIR / "results.json").write_text(
        json.dumps({"updated": stamp,
                    "played": len(results_payload),
                    "with_prediction": rated,
                    "correct": correct,
                    "results": results_payload},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Ergebnisse: {OUTDIR/'results.json'} ({len(results_payload)} gespielt, "
          f"{correct}/{rated} Tendenz richtig)")


if __name__ == "__main__":
    main()
