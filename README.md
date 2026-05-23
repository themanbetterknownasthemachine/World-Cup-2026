# WM 2026 Prognose-Modell

Machine-Learning-Projekt zur probabilistischen Vorhersage der Fussball-WM 2026
(48 Teams, 12 Vierergruppen, 104 Spiele, K.-o.-Phase mit 32 Teams).

Das Ziel ist **nicht**, einzelne Resultate exakt zu treffen — das ist im Fussball
kaum möglich — sondern **Wahrscheinlichkeiten** zu schaetzen: pro Spiel und am Ende
fuer den Titelgewinn jedes Teams.

## Der Ansatz in drei Ebenen

| Ebene | Was | Wie |
|-------|-----|-----|
| **A** | Sieg / Remis / Niederlage pro Spiel (z.B. 47% / 13% / 40%) | Dixon-Coles-Tor-Modell |
| **B** | Exaktes Resultat (2:1, 0:0 …) als Zwischenschritt fuer Tordifferenzen | Score-Matrix aus Ebene A |
| **C** | Wer wird Weltmeister? (z.B. Argentinien 14%) | Monte-Carlo-Simulation des Turniers |

Der rote Faden: **Ebene A liefert die Bausteine → Monte Carlo aggregiert sie zu Ebene C.**
Eine spaetere Ausbaustufe ersetzt den Dixon-Coles-Motor in Ebene A durch ein
ML-Modell (LightGBM mit Elo-Features) — die restliche Pipeline bleibt identisch.

## Projektstruktur

```
wm2026/
├── README.md                          # diese Datei
├── requirements.txt                   # Python-Abhaengigkeiten
├── .gitignore
├── data/                              # Rohdaten (nicht im Git, s. .gitignore)
├── docs/
│   └── methodik.md                    # Theorie: Poisson, Dixon-Coles, RPS, Monte Carlo
└── notebooks/
    └── 01_dixon_coles_baseline.ipynb  # Baseline-Modell + WM-2022-Backtest
```

## Setup

```bash
# 1. Virtuelle Umgebung (empfohlen)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Abhaengigkeiten
pip install -r requirements.txt

# 3. Notebook starten
jupyter lab notebooks/01_dixon_coles_baseline.ipynb
```

Die Trainingsdaten (internationale Spielresultate seit 1872) zieht das Notebook
automatisch vom oeffentlichen GitHub-Mirror des Kaggle-Datensatzes
`martj42/international-football-results`. Optional kannst du `results.csv` manuell
nach `data/` legen.

## Roadmap

- [x] **Phase 1** — Dixon-Coles-Baseline, Backtest auf WM 2022 (dieses Notebook)
- [ ] **Phase 2** — Monte-Carlo-Simulator fuer das 48-Team-Format der WM 2026
- [ ] **Phase 3** — TheSportsDB-Live-Schicht (Spielplan + laufende Resultate)
- [ ] **Phase 4** — Ebene-A-Upgrade: LightGBM mit Elo-/Form-Features, Vergleich per RPS
