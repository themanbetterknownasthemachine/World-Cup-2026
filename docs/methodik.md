# Methodik

Dieses Dokument erklaert die statistischen Grundlagen hinter dem Modell — bewusst
mit Intuition statt nur Formeln.

## 1. Warum Wahrscheinlichkeiten statt Resultaten?

Fussball ist tor-arm und damit stark zufallsbehaftet. Selbst die besten Modelle
treffen die richtige Tendenz (1/X/2) nur in rund 50–55% der Spiele. Eine seriöse
"Vorhersage" ist deshalb nie "Heimsieg!", sondern eine Verteilung wie
*Heim 47% / Remis 13% / Auswaerts 40%* — genau das, was im TV vor dem Anpfiff
eingeblendet wird (das ist unsere **Ebene A**).

## 2. Poisson- vs. Normalverteilung

Der Unterschied liegt in der Art der Groesse:

- **Normalverteilung** — fuer *kontinuierliche, symmetrische* Messgroessen, die um
  einen Mittelwert streuen (Koerpergroesse, Temperatur, Messfehler). Glockenkurve,
  erlaubt Kommazahlen und negative Werte.
- **Poisson-Verteilung** — fuer die *Anzahl* relativ seltener Ereignisse in einem
  festen Intervall (Tore pro Spiel, Anrufe pro Stunde, Defekte pro Charge). Nur
  ganzzahlige Werte ≥ 0, rechtsschief, definiert durch **einen** Parameter λ, der
  gleichzeitig Mittelwert *und* Varianz ist.

Tore sind ganzzahlige Zaehlereignisse (kein halbes, kein negatives Tor), treten
selten und annaehernd unabhaengig ueber 90 Minuten auf, Schnitt ~1.3 pro Team.
Eine Normalverteilung wuerde Wahrscheinlichkeitsmasse auf "−1.4 Tore" legen — Unsinn.
**Faustregel: misst du etwas → Normal; zaehlst du etwas Seltenes → Poisson.**

## 3. Das Dixon-Coles-Modell

Jedes Team bekommt zwei Kennzahlen, die aus allen vergangenen Spielen gleichzeitig
per Maximum Likelihood geschaetzt werden:

- **Angriffsstaerke** `att` — wie viele Tore schiesst es typischerweise
- **Abwehrstaerke** `def` — wie viele laesst es zu

Die erwartete Toranzahl je Team in einem Spiel (Heim *i* gegen Auswaerts *j*):

```
λ (Heimtore)     = exp(att_i + def_j + Heimvorteil)
μ (Auswaertstore) = exp(att_j + def_i)
```

Bei neutralem Platz (fast alle WM-Spiele) faellt der Heimvorteil weg. Aus λ und μ
macht die Poisson-Verteilung die Wahrscheinlichkeit fuer jede Toranzahl — kombiniert
ergibt sich eine **Score-Matrix** `P(Heim schiesst x, Auswaerts schiesst y)`. Daraus
faellt alles ab: 1/X/2, Ueber/Unter 2.5, exakte Resultate.

### Die zwei Dixon-Coles-Ergänzungen zu reinem Poisson

1. **Niedrig-Resultat-Korrektur (τ, "tau"):** Knappe Ergebnisse (0:0, 1:0, 0:1, 1:1)
   sind in echt etwas häufiger als reines Poisson behauptet. Ein Korrekturfaktor mit
   Parameter ρ ("rho") biegt genau diese vier Zellen der Score-Matrix gerade.

2. **Zeitgewichtung:** Aeltere Spiele zaehlen weniger, weil Teamstaerke ueber Jahre
   driftet. Gewicht eines Spiels = `exp(-ξ · Tage_vor_Stichtag)`. Der Parameter ξ
   ("xi") steuert die Halbwertszeit — `ξ = 0.0018/Tag` entspricht ~1 Jahr. Das ist
   der wichtigste Tuning-Parameter.

### Identifizierbarkeit

`att_i + def_j` aendert sich nicht, wenn man zu allen Angriffswerten eine Konstante
addiert und von allen Abwehrwerten abzieht. Diese Mehrdeutigkeit fixieren wir, indem
wir die Angriffswerte auf Mittelwert 0 zentrieren. (Deshalb meldet der Optimierer
"converged: False" — die zentrierte Richtung ist flach. Das ist harmlos, die
relevanten Parameter sind stabil.)

## 4. Bewertung: Ranked Probability Score (RPS)

Fuer probabilistische Prognosen geordneter Klassen (Heim < Remis < Auswaerts) ist
der RPS die Standardmetrik — das Pendant zu MAPE/MAE, aber fuer Wahrscheinlichkeiten.

Mit Prognose `p = [pH, pD, pA]` und Ausgang als One-Hot `o`:

```
RPS = 1/(r-1) · Σ (kumP_k - kumO_k)²        # r = 3 Klassen → Faktor 1/2
```

Er bestraft, *wie weit daneben* eine Prognose liegt: einen Auswaertssieg als "Remis"
vorherzusagen ist weniger schlimm als ihn als "klaren Heimsieg" vorherzusagen.
**Niedriger = besser.** Gut kalibrierte WM-Modelle liegen bei ~0.18–0.22, eine naive
Konstant-Prognose bei ~0.23.

> Hinweis WM 2022: Das war ein extrem ueberraschungsreiches Turnier. Ein dünner
> Vorsprung der Baseline gegenueber "naiv" ist dort normal und kein Modellfehler —
> bei einem einzelnen Turnier dominiert der Zufall. Genau deshalb simulieren wir die
> WM 2026 spaeter tausendfach (Monte Carlo), statt einem Lauf zu vertrauen.

## 5. Ausblick: Monte-Carlo-Simulation (Ebene C)

Die Titelfrage hat keine geschlossene Formel — das Turnier verzweigt ueber 104 Spiele,
jeder K.-o.-Gegner haengt von vorherigen Resultaten ab. Loesung: das komplette Turnier
z.B. 10'000-mal durchsimulieren, in jedem Lauf jedes Spiel anhand seiner
Score-Matrix auswuerfeln, Gruppen + die fiese "8 beste Gruppendritte"-Logik + Bracket
durchspielen — und am Ende zaehlen, wie oft jedes Team gewinnt. Die Haeufigkeiten
konvergieren gegen die wahren Titelwahrscheinlichkeiten (Gesetz der grossen Zahlen).
