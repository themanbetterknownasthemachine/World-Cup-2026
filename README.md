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
├── .gitignore
├── data/                                  # Rohdaten (nicht im Git, s. .gitignore)
│   └── results.csv                        # wird beim ersten Lauf automatisch gecached
├── docs/
│   └── methodik.md                        # Theorie: Poisson, Dixon-Coles, RPS, Monte Carlo
└── notebooks/
    ├── 01_dixon_coles_baseline.ipynb      # Phase 1: Baseline-Modell + WM-2022-Backtest
    ├── 02_monte_carlo_simulation.ipynb    # Phase 2: 48-Team-Turnier-Simulation
    └── 03_elo_model.ipynb                 # Phase 4: Elo-Engine als Ebene-A-Upgrade
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

## Roadmap

- [x] **Phase 1** — Dixon-Coles-Baseline, Backtest auf WM 2022 ([01_dixon_coles_baseline.ipynb](notebooks/01_dixon_coles_baseline.ipynb))
- [x] **Phase 2** — Monte-Carlo-Simulator fuer das 48-Team-Format der WM 2026 ([02_monte_carlo_simulation.ipynb](notebooks/02_monte_carlo_simulation.ipynb))
- [x] **Phase 4** — Ebene-A-Upgrade auf Elo-Engine, +4.4% RPS-Verbesserung gegenueber DC-Baseline ([03_elo_model.ipynb](notebooks/03_elo_model.ipynb))
- [ ] **Phase 2b** — offizielles FIFA-Bracket statt Finish-Seeding (inkl. Zuordnungstabelle der 8 besten Gruppendritten)
- [ ] **Phase 3** — TheSportsDB-Live-Schicht (Spielplan + laufende Resultate, taegliche Re-Simulation)
- [ ] **Phase 5** — LightGBM mit Zusatz-Features (Ruhetage, Reisedistanz, Kaderwert), erneut per RPS gegen Elo messen
