# WM 2026 Prognose-Modell

Machine-Learning-Projekt zur probabilistischen Vorhersage der Fussball-WM 2026
(48 Teams, 12 Vierergruppen, 104 Spiele, K.-o.-Phase mit 32 Teams).

Das Ziel ist **nicht**, einzelne Resultate exakt zu treffen — das ist im Fussball
kaum möglich — sondern **Wahrscheinlichkeiten** zu schaetzen: pro Spiel und am Ende
fuer den Titelgewinn jedes Teams.

## Der Ansatz in drei Ebenen

| Ebene | Was | Wie |
|-------|-----|-----|
| **A** | Sieg / Remis / Niederlage pro Spiel (z.B. 47% / 13% / 40%) | Dixon-Coles-Tor-Modell → Elo-Upgrade |
| **B** | Exaktes Resultat (2:1, 0:0 …) als Zwischenschritt fuer Tordifferenzen | Score-Matrix aus Ebene A |
| **C** | Wer wird Weltmeister? (z.B. Argentinien 14%) | Monte-Carlo-Simulation des Turniers |

Der rote Faden: **Ebene A liefert die Bausteine → Monte Carlo aggregiert sie zu Ebene C.**
In Notebook 03 wurde der Dixon-Coles-Motor in Ebene A durch eine **Elo-basierte
Engine** ersetzt (RPS 0.2215 statt 0.2316 im WC-2022-Backtest) — die restliche
Pipeline bleibt identisch.

## Projektstruktur

```
wm2026/
├── README.md                              # diese Datei
├── requirements.txt                       # Python-Abhaengigkeiten
├── .github/workflows/
│   └── forecast.yml                       # GitHub-Actions-Workflow (taegliche Live-Prognose)
├── .gitignore
├── data/                                  # Rohdaten (nicht im Git, s. .gitignore)
│   └── results.csv                        # wird beim ersten Lauf automatisch gecached
├── docs/
│   ├── methodik.md                        # Theorie: Poisson, Dixon-Coles, RPS, Monte Carlo
│   ├── index.html                         # statisches Live-Dashboard (Chart.js + Vanilla JS)
│   └── data/                              # Spiegel von output/ fuer GitHub Pages
├── notebooks/
│   ├── 01_dixon_coles_baseline.ipynb      # Phase 1: Baseline-Modell + WM-2022-Backtest
│   ├── 02_monte_carlo_simulation.ipynb    # Phase 2: 48-Team-Turnier-Simulation
│   ├── 03_elo_model.ipynb                 # Phase 4: Elo-Engine als Ebene-A-Upgrade
│   ├── 04_live_update.ipynb               # Phase 3: Live-Schicht (Trockentest + Erklaerung)
│   └── 05_match_predictions.ipynb         # Pro-Spiel-Prognose (1/X/2 + wahrscheinlichstes Resultat, interaktiv)
├── src/                                   # geteilte Kernlogik (Notebook + Automatik)
│   ├── wm_model.py                        # Elo + Tor-Modell + Gruppen + Simulation + Pro-Spiel-Prognose
│   └── live_forecast.py                   # Headless-Skript fuer die taegliche Automatik
└── output/                                # generierte Prognose (von der Automatik gepflegt)
    ├── forecast.json / forecast.md        # Titel-/Finale-/Halbfinal-Wahrscheinlichkeiten (Ebene C)
    ├── matches.json  / matches.md         # Pro-Spiel-Prognose 1/X/2 + wahrscheinlichstes Resultat (Ebene A)
    ├── results.json                       # gespielte Spiele + Pre-Match-Prognose (Dashboard-Vergleich)
    ├── predictions_archive/               # eingefrorene Pre-Match-Snapshots pro Spieltag
    ├── history.csv                        # Zeitreihe der Titelchancen (eine Zeile je Team und Tag)
    └── goal_coefs.json                    # gecachte Tor-Modell-Koeffizienten (vermeidet Re-Fit)
```

## Setup (conda)

```powershell
# 1. Conda-Umgebung anlegen (Python 3.12)
conda create -n wm2026 python=3.12 -y
conda activate wm2026

# 2. Abhaengigkeiten
pip install -r requirements.txt

# 3. Jupyter-Kernel registrieren (damit VS Code / JupyterLab das env sieht)
python -m ipykernel install --user --name wm2026 --display-name "Python (wm2026)"

# 4. Notebook starten
jupyter lab notebooks/01_dixon_coles_baseline.ipynb
```

In VS Code: oben rechts im Notebook den Kernel auf **"Python (wm2026)"** stellen.

## Daten

Die Trainingsdaten (internationale Spielresultate seit 1872) zieht das Notebook
beim ersten Lauf automatisch vom oeffentlichen GitHub-Mirror des Kaggle-Datensatzes
`martj42/international-football-results` und legt sie als `data/results.csv` ab —
ab dann wird lokal geladen, ohne Netz. Bei Verbindungsproblemen retryt der Loader
bis zu 3-mal. Du kannst `results.csv` auch manuell nach `data/` legen.

## Live-Prognose & Automatik

Waehrend des Turniers wird die Prognose taeglich aktualisiert: gespielte
WM-Resultate von **TheSportsDB** fliessen in die Elo-Wertung ein, gespielte Partien
werden in der Simulation fixiert, der Rest neu durchgewuerfelt.

Manuell starten:

```powershell
$env:SPORTSDB_KEY = "DEIN_PATREON_KEY"   # ohne Key: Vorturnier-Prognose, mit Warnhinweis
python src/live_forecast.py              # schreibt output/forecast.{json,md}, matches.{json,md}, history.csv
```

Pro Lauf entstehen vier Artefakte:

- `forecast.json` / `forecast.md` — Titel-/Finale-/Halbfinal-Wahrscheinlichkeiten (Ebene C)
- `matches.json` / `matches.md` — Pro-Spiel-Prognose mit 1/X/2 + wahrscheinlichstem Resultat (Ebene A)
- `history.csv` — Zeitreihe: pro Tag eine Zeile je Team, mehrfache Laeufe am selben UTC-Tag ueberschreiben sich
- `goal_coefs.json` — Cache der Tor-Modell-Koeffizienten, damit nicht jeder Lauf neu fittet

Automatisch: [`.github/workflows/forecast.yml`](.github/workflows/forecast.yml) ist
ein **GitHub-Actions-Workflow**, der das Skript taeglich um 06:00 UTC laufen
laesst und das Ergebnis zurueck ins Repo committet. API-Key in den Repo-Settings
als Secret `SPORTSDB_KEY` hinterlegen.

Notebook 04 ist der interaktive Trockentest derselben Live-Logik; Notebook 05
zeigt die Pro-Spiel-Prognose mit Filtern (nach Team, Datum, „spannendste Spiele")
— alle drei nutzen `src/wm_model.py` als geteilte Engine.

## Dashboard (GitHub Pages)

`docs/index.html` ist ein statisches Single-Page-Dashboard, das die JSON-Artefakte
direkt im Browser rendert — kein Server, kein Login, einfach bookmarken. Sektionen:

- **Pulse** — Countdown bis Anpfiff, Anzahl beruecksichtigter Spiele, aktueller Favorit, Tendenz-Trefferquote des Modells
- **Titel-Chancen Top 15** — gestaffelte Balken (Titel / Finale / Halbfinale)
- **Verlauf Top 6** — Linienchart der Titel-Chance ueber alle bisherigen Cron-Laeufe (`history.csv`)
- **Naechste 8 Spiele** — 1/X/2-Wahrscheinlichkeiten + wahrscheinlichstes Resultat
- **Prognose vs. Realitaet** — gespielte Spiele mit dem Forecast, der vor Anpfiff im `predictions_archive/` eingefroren wurde, inklusive ✓/✗ und "SCHOCK"-Badge bei Underdog-Siegen
- **Bewegungs-Spotlight** — Tagessieger/-verlierer in der Titel-Chance

### Lokal anschauen

```powershell
python -m http.server 8765 -d docs
# Browser: http://127.0.0.1:8765/
```
Reines `file://` funktioniert nicht (CORS auf `fetch`), deshalb der Mini-Server.

### Auf GitHub Pages aktivieren (einmalig)

1. Repo → **Settings → Pages**
2. Source: **Deploy from a branch**
3. Branch: `main` / Folder: `/docs` → **Save**
4. Nach ein paar Minuten ist die Seite unter `https://<dein-user>.github.io/wm2026/` erreichbar
5. Der taegliche Workflow spiegelt `output/*.json` + `history.csv` nach `docs/data/` und committet — das Dashboard ist also nach jedem 06:00-UTC-Lauf frisch

## Roadmap

- [x] **Phase 1** — Dixon-Coles-Baseline, Backtest auf WM 2022 ([01_dixon_coles_baseline.ipynb](notebooks/01_dixon_coles_baseline.ipynb))
- [x] **Phase 2** — Monte-Carlo-Simulator fuer das 48-Team-Format der WM 2026 ([02_monte_carlo_simulation.ipynb](notebooks/02_monte_carlo_simulation.ipynb))
- [x] **Phase 4** — Ebene-A-Upgrade auf Elo-Engine, +4.4% RPS-Verbesserung gegenueber DC-Baseline ([03_elo_model.ipynb](notebooks/03_elo_model.ipynb))
- [x] **Phase 3** — TheSportsDB-Live-Schicht, taegliche Re-Simulation via GitHub Actions ([04_live_update.ipynb](notebooks/04_live_update.ipynb), [src/](src/), [output/](output/))
- [ ] **Phase 2b** — offizielles FIFA-Bracket statt Finish-Seeding (inkl. Zuordnungstabelle der 8 besten Gruppendritten)
- [ ] **Phase 5** — LightGBM mit Zusatz-Features (Ruhetage, Reisedistanz, Kaderwert), erneut per RPS gegen Elo messen
