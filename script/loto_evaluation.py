#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Évaluation scientifique des méthodes de prédiction Loto
=======================================================

Ce module répond à une seule question, honnêtement et avec des chiffres :

    « Est-ce qu'une méthode de "prédiction" bat réellement le hasard
      sur le Loto français (5 numéros / 49 + 1 Chance / 10) ? »

Il fournit trois briques complémentaires :

1.  Les **cotes exactes** de chaque rang (calcul hypergéométrique, pas
    d'estimation), donc la probabilité réelle de gagner.
2.  L'**espérance de gain** (et le taux de retour au joueur, RTP) à partir
    d'une grille de gains paramétrable.
3.  Un **backtest walk-forward** qui rejoue l'histoire tirage par tirage :
    pour chaque tirage, chaque stratégie ne voit que le passé, propose une
    grille, et on la compare au résultat réel. Les stratégies sont ensuite
    confrontées à une **référence Monte-Carlo aléatoire** avec un test de
    significativité.

Résultat attendu, et c'est tout l'intérêt de pouvoir le mesurer : aucune
stratégie ne se distingue significativement du hasard, parce que des tirages
indépendants et équiprobables ne contiennent aucun signal à apprendre.

Dépendances : numpy, pandas (déjà dans requirements.txt). Aucune dépendance
lourde (scikit-learn n'est pas nécessaire pour évaluer la prédictibilité).

Usage :
    python script/loto_evaluation.py --csv tirages_loto.csv
    python script/loto_evaluation.py --csv tirages_loto.csv --mc-runs 500 --json rapport.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paramètres du Loto français (constants depuis la réforme du 6 octobre 2008)
# ---------------------------------------------------------------------------
MAIN_MAX: int = 49          # numéros principaux : 1..49
MAIN_PICK: int = 5          # 5 numéros tirés / joués
CHANCE_MAX: int = 10        # numéro Chance : 1..10
TICKET_PRICE: float = 2.20  # prix d'une grille simple (EUR)

NUMBER_COLS = [f"Numéro {i}" for i in range(1, MAIN_PICK + 1)]
CHANCE_COL = "Chance"


# ---------------------------------------------------------------------------
# 1. Cotes exactes (combinatoire, aucune approximation)
# ---------------------------------------------------------------------------
def total_combinations() -> int:
    """Nombre total de grilles distinctes : C(49,5) * 10."""
    return math.comb(MAIN_MAX, MAIN_PICK) * CHANCE_MAX


def prob_k_main_matches(k: int) -> float:
    """
    Probabilité d'avoir exactement k bons numéros principaux (loi
    hypergéométrique). On choisit 5 numéros ; 5 sont "gagnants" parmi 49.
    """
    if k < 0 or k > MAIN_PICK:
        return 0.0
    favorable = math.comb(MAIN_PICK, k) * math.comb(MAIN_MAX - MAIN_PICK, MAIN_PICK - k)
    return favorable / math.comb(MAIN_MAX, MAIN_PICK)


# Rangs gagnants officiels du Loto : (bons numéros, chance gagnée)
WINNING_RANKS: List[Tuple[str, int, int]] = [
    ("5 + Chance", 5, 1),
    ("5", 5, 0),
    ("4 + Chance", 4, 1),
    ("4", 4, 0),
    ("3 + Chance", 3, 1),
    ("3", 3, 0),
    ("2 + Chance", 2, 1),
    ("2", 2, 0),
    ("1 + Chance", 1, 1),
    ("0 + Chance", 0, 1),
]


def rank_probabilities() -> Dict[str, float]:
    """Probabilité exacte de chaque rang gagnant pour une grille simple."""
    p_chance = 1.0 / CHANCE_MAX
    out: Dict[str, float] = {}
    for label, k, need_chance in WINNING_RANKS:
        p_main = prob_k_main_matches(k)
        out[label] = p_main * (p_chance if need_chance else (1.0 - p_chance))
    return out


def prob_any_prize() -> float:
    """Probabilité de gagner au moins quelque chose (n'importe quel rang)."""
    return sum(rank_probabilities().values())


# Grille de gains *indicative* (EUR). Les rangs élevés sont en réalité
# parimutuels (ils varient à chaque tirage) ; ces montants sont des ordres de
# grandeur publics servant au calcul d'espérance. Les montants réels ne
# changent pas la conclusion : le RTP reste largement inférieur à 1.
DEFAULT_PRIZES: Dict[str, float] = {
    "5 + Chance": 2_000_000.0,  # jackpot minimum garanti (variable, souvent +)
    "5": 100_000.0,             # variable
    "4 + Chance": 1_000.0,      # variable
    "4": 400.0,                 # variable
    "3 + Chance": 50.0,         # variable
    "3": 20.0,                  # variable
    "2 + Chance": 10.0,         # variable
    "2": 4.40,                  # ~fixe
    "1 + Chance": 2.20,         # ~remboursement
    "0 + Chance": 2.20,         # remboursement
}


def expected_value(prizes: Optional[Dict[str, float]] = None,
                   ticket_price: float = TICKET_PRICE) -> Dict[str, float]:
    """
    Espérance de gain d'une grille simple.

    Returns dict avec gain espéré brut, espérance nette (gain - mise) et RTP
    (return-to-player = gain espéré / mise).
    """
    prizes = prizes or DEFAULT_PRIZES
    probs = rank_probabilities()
    gross = sum(probs[label] * prizes.get(label, 0.0) for label in probs)
    return {
        "gain_espere_brut": gross,
        "esperance_nette": gross - ticket_price,
        "rtp": gross / ticket_price if ticket_price else float("nan"),
        "prix_grille": ticket_price,
    }


def breakeven_jackpot(prizes: Optional[Dict[str, float]] = None,
                      ticket_price: float = TICKET_PRICE) -> float:
    """
    Jackpot (rang 5+Chance) qui rendrait l'espérance BRUTE égale à la mise,
    tous les autres rangs restant à leur valeur de référence.

    Nuance honnête et importante : au-dessus de ce seuil, l'espérance brute
    d'une grille dépasse son prix. C'est réel (c'est le principe des rares cas
    d'« arbitrage » de loterie), mais ce n'est PAS une capacité de prédiction,
    et ce n'est pas une martingale exploitable pour un joueur, car :
      - le jackpot se PARTAGE entre tous les gagnants (dilution) ;
      - la variance est colossale (il faut ~des millions de grilles pour que
        la loi des grands nombres joue, donc un capital énorme) ;
      - fiscalité, plafonds de mise et règles de rollover s'y opposent.
    Le seuil est volontairement calculé pour montrer à quel point il est élevé.
    """
    prizes = dict(prizes or DEFAULT_PRIZES)
    probs = rank_probabilities()
    p_jackpot = probs["5 + Chance"]
    gross_others = sum(
        probs[label] * prizes.get(label, 0.0)
        for label in probs if label != "5 + Chance"
    )
    return (ticket_price - gross_others) / p_jackpot


# ---------------------------------------------------------------------------
# 2. Chargement des données
# ---------------------------------------------------------------------------
@dataclass
class DrawData:
    """Tirages chargés sous forme de tableaux numpy."""
    main: np.ndarray    # shape (N, 5), int
    chance: np.ndarray  # shape (N,), int

    def __len__(self) -> int:
        return self.main.shape[0]


def load_draws(csv_path: str | Path) -> DrawData:
    """Charge et valide le CSV des tirages."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    missing = [c for c in NUMBER_COLS + [CHANCE_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    main = df[NUMBER_COLS].to_numpy(dtype=float)
    chance = df[CHANCE_COL].to_numpy(dtype=float)

    # Lignes complètes et dans les bornes uniquement.
    valid = (
        ~np.isnan(main).any(axis=1)
        & ~np.isnan(chance)
        & (main >= 1).all(axis=1) & (main <= MAIN_MAX).all(axis=1)
        & (chance >= 1) & (chance <= CHANCE_MAX)
    )
    return DrawData(main=main[valid].astype(int), chance=chance[valid].astype(int))


# ---------------------------------------------------------------------------
# 3. Stratégies de "prédiction"
#    Signature commune : (hist_main, hist_chance, rng) -> (set[int] de 5, int)
#    Chacune ne voit QUE l'historique passé.
# ---------------------------------------------------------------------------
Strategy = Callable[[np.ndarray, np.ndarray, np.random.Generator], Tuple[set, int]]


def _frequencies(hist_main: np.ndarray, hist_chance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compte d'apparition de chaque numéro principal (1..49) et chance (1..10)."""
    main_counts = np.bincount(hist_main.ravel(), minlength=MAIN_MAX + 1)[1:]
    chance_counts = np.bincount(hist_chance.ravel(), minlength=CHANCE_MAX + 1)[1:]
    return main_counts, chance_counts


def strat_random(hist_main, hist_chance, rng) -> Tuple[set, int]:
    """Référence : grille purement aléatoire, uniforme."""
    nums = set(rng.choice(np.arange(1, MAIN_MAX + 1), size=MAIN_PICK, replace=False).tolist())
    ch = int(rng.integers(1, CHANCE_MAX + 1))
    return nums, ch


def strat_hot(hist_main, hist_chance, rng) -> Tuple[set, int]:
    """Numéros les plus fréquents ("hot numbers")."""
    main_counts, chance_counts = _frequencies(hist_main, hist_chance)
    nums = set((np.argsort(main_counts)[::-1][:MAIN_PICK] + 1).tolist())
    ch = int(np.argmax(chance_counts) + 1)
    return nums, ch


def strat_cold(hist_main, hist_chance, rng) -> Tuple[set, int]:
    """Numéros les moins fréquents ("cold numbers" / retardataires)."""
    main_counts, chance_counts = _frequencies(hist_main, hist_chance)
    nums = set((np.argsort(main_counts)[:MAIN_PICK] + 1).tolist())
    ch = int(np.argmin(chance_counts) + 1)
    return nums, ch


def strat_recency(hist_main, hist_chance, rng, half_life: int = 50) -> Tuple[set, int]:
    """Pondération par récence : les tirages récents pèsent plus."""
    n = hist_main.shape[0]
    weights = np.exp(-(np.arange(n)[::-1]) / half_life)  # plus récent -> poids fort
    main_score = np.zeros(MAIN_MAX + 1)
    chance_score = np.zeros(CHANCE_MAX + 1)
    for i in range(n):
        for num in hist_main[i]:
            main_score[num] += weights[i]
        chance_score[hist_chance[i]] += weights[i]
    nums = set((np.argsort(main_score[1:])[::-1][:MAIN_PICK] + 1).tolist())
    ch = int(np.argmax(chance_score[1:]) + 1)
    return nums, ch


def strat_fibonacci(hist_main, hist_chance, rng) -> Tuple[set, int]:
    """
    Pondération "Fibonacci" (l'idée du projet) : on classe les numéros par
    fréquence puis on applique des poids de Fibonacci au classement. Inclus
    pour pouvoir mesurer qu'il ne bat pas non plus le hasard.
    """
    main_counts, chance_counts = _frequencies(hist_main, hist_chance)
    fib = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
    order = np.argsort(main_counts)[::-1]  # du plus fréquent au moins fréquent
    weights = np.zeros(MAIN_MAX)
    for rank, idx in enumerate(order):
        weights[idx] = fib[rank % len(fib)]
    nums = set((np.argsort(weights)[::-1][:MAIN_PICK] + 1).tolist())
    ch = int(np.argmax(chance_counts) + 1)
    return nums, ch


STRATEGIES: Dict[str, Strategy] = {
    "aleatoire (reference)": strat_random,
    "frequents (hot)": strat_hot,
    "retardataires (cold)": strat_cold,
    "recence": strat_recency,
    "fibonacci": strat_fibonacci,
}


# ---------------------------------------------------------------------------
# 4. Backtest walk-forward
# ---------------------------------------------------------------------------
@dataclass
class StrategyResult:
    name: str
    n_draws: int
    mean_main_matches: float
    std_main_matches: float
    chance_hit_rate: float
    prize_rate: float                      # part des grilles gagnant un rang
    rank_counts: Dict[str, int] = field(default_factory=dict)
    z_vs_null: float = 0.0                  # z-score vs hasard théorique
    p_value: float = 1.0                    # bilatéral, approx normale


# Moments théoriques du nombre de bons numéros (loi hypergéométrique)
NULL_MEAN_MATCHES = MAIN_PICK * MAIN_PICK / MAIN_MAX  # 25/49 ≈ 0.5102
NULL_VAR_MATCHES = (
    MAIN_PICK
    * (MAIN_PICK / MAIN_MAX)
    * ((MAIN_MAX - MAIN_PICK) / MAIN_MAX)
    * ((MAIN_MAX - MAIN_PICK) / (MAIN_MAX - 1))
)


def _rank_of(matched: int, chance_hit: bool) -> Optional[str]:
    for label, k, need_chance in WINNING_RANKS:
        if matched == k and (need_chance == (1 if chance_hit else 0)):
            return label
    return None


def walk_forward(draws: DrawData, strategy: Strategy, name: str,
                 min_history: int, rng: np.random.Generator) -> StrategyResult:
    """Rejoue l'histoire : prédire le tirage i à partir des tirages [0:i]."""
    n = len(draws)
    matches: List[int] = []
    chance_hits = 0
    rank_counts: Dict[str, int] = {}

    for i in range(min_history, n):
        hist_main = draws.main[:i]
        hist_chance = draws.chance[:i]
        pred_nums, pred_chance = strategy(hist_main, hist_chance, rng)

        actual_nums = set(draws.main[i].tolist())
        matched = len(pred_nums & actual_nums)
        chance_hit = (pred_chance == int(draws.chance[i]))

        matches.append(matched)
        chance_hits += int(chance_hit)
        rank = _rank_of(matched, chance_hit)
        if rank:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

    matches_arr = np.array(matches, dtype=float)
    m = len(matches_arr)
    mean_matches = float(matches_arr.mean())
    prize_rate = sum(rank_counts.values()) / m if m else 0.0

    # Test de significativité : l'écart à la moyenne théorique du hasard
    # est-il explicable par le simple bruit d'échantillonnage ?
    se = math.sqrt(NULL_VAR_MATCHES / m) if m else float("nan")
    z = (mean_matches - NULL_MEAN_MATCHES) / se if se else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))  # p-value bilatérale (approx normale)

    return StrategyResult(
        name=name,
        n_draws=m,
        mean_main_matches=mean_matches,
        std_main_matches=float(matches_arr.std(ddof=1)) if m > 1 else 0.0,
        chance_hit_rate=chance_hits / m if m else 0.0,
        prize_rate=prize_rate,
        rank_counts=dict(sorted(rank_counts.items())),
        z_vs_null=z,
        p_value=p,
    )


def monte_carlo_null(draws: DrawData, min_history: int, n_runs: int,
                     rng: np.random.Generator) -> Dict[str, float]:
    """
    Référence empirique : distribution de la moyenne de bons numéros obtenue
    par des grilles aléatoires, répétée n_runs fois. Donne une "barre de
    bruit" à laquelle comparer les stratégies.
    """
    n = len(draws)
    m = n - min_history
    run_means = np.empty(n_runs)
    actual = draws.main[min_history:]  # (m, 5)
    all_nums = np.arange(1, MAIN_MAX + 1)
    for r in range(n_runs):
        total = 0
        for j in range(m):
            pick = set(rng.choice(all_nums, size=MAIN_PICK, replace=False).tolist())
            total += len(pick & set(actual[j].tolist()))
        run_means[r] = total / m
    return {
        "mc_mean": float(run_means.mean()),
        "mc_std": float(run_means.std(ddof=1)) if n_runs > 1 else 0.0,
        "mc_p05": float(np.percentile(run_means, 5)),
        "mc_p95": float(np.percentile(run_means, 95)),
        "n_runs": n_runs,
    }


# ---------------------------------------------------------------------------
# 5. Rapport
# ---------------------------------------------------------------------------
def build_report(csv_path: str | Path, min_history: int = 200,
                 mc_runs: int = 300, seed: int = 12345) -> Dict:
    draws = load_draws(csv_path)
    if len(draws) < min_history + 10:
        raise ValueError(
            f"Trop peu de tirages ({len(draws)}) pour min_history={min_history}."
        )

    rng = np.random.default_rng(seed)

    odds = rank_probabilities()
    ev = expected_value()

    results: List[StrategyResult] = []
    for name, strat in STRATEGIES.items():
        results.append(walk_forward(draws, strat, name, min_history, rng))

    mc = monte_carlo_null(draws, min_history, mc_runs, rng)

    return {
        "dataset": {
            "csv": str(csv_path),
            "n_draws_total": len(draws),
            "min_history": min_history,
            "n_draws_evaluated": len(draws) - min_history,
        },
        "cotes_exactes": {
            "combinaisons_totales": total_combinations(),
            "prob_rangs": odds,
            "prob_gagner_quelque_chose": prob_any_prize(),
            "chance_contre_1_jackpot": round(1 / odds["5 + Chance"]),
        },
        "esperance": ev,
        "backtest": [asdict(r) for r in results],
        "reference_aleatoire_monte_carlo": mc,
        "moyenne_theorique_bons_numeros": NULL_MEAN_MATCHES,
    }


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.4f}%"


def print_report(report: Dict) -> None:
    ds = report["dataset"]
    odds = report["cotes_exactes"]
    ev = report["esperance"]
    mc = report["reference_aleatoire_monte_carlo"]

    print("=" * 74)
    print(" ÉVALUATION SCIENTIFIQUE DE LA PRÉDICTIBILITÉ DU LOTO")
    print("=" * 74)
    print(f" Données : {ds['n_draws_total']} tirages | "
          f"{ds['n_draws_evaluated']} évalués (historique min. {ds['min_history']})")
    print()

    print("-- Cotes exactes (une grille simple) " + "-" * 36)
    print(f"  Combinaisons possibles        : {odds['combinaisons_totales']:,}".replace(",", " "))
    print(f"  Probabilité du jackpot (5+Ch) : 1 sur {odds['chance_contre_1_jackpot']:,}".replace(",", " "))
    print(f"  Gagner au moins un rang       : {_fmt_pct(odds['prob_gagner_quelque_chose'])} "
          f"(~1 sur {round(1 / odds['prob_gagner_quelque_chose'])})")
    print()

    print("-- Espérance de gain (grille de gains indicative) " + "-" * 24)
    print(f"  Prix de la grille   : {ev['prix_grille']:.2f} EUR")
    print(f"  Gain espéré         : {ev['gain_espere_brut']:.3f} EUR")
    print(f"  Espérance nette     : {ev['esperance_nette']:.3f} EUR par grille")
    print(f"  Taux de retour (RTP): {ev['rtp'] * 100:.1f}%  "
          f"(< 100% => perte moyenne garantie à long terme)")
    be = breakeven_jackpot()
    print(f"  Seuil d'EV nulle     : jackpot ~{be / 1e6:.1f} M EUR "
          f"(au-delà, EV brute > mise, mais partage + variance => pas une martingale)")
    print()

    print("-- Backtest walk-forward : bons numéros par grille " + "-" * 23)
    print(f"  Référence théorique du pur hasard : {report['moyenne_theorique_bons_numeros']:.4f} bons/grille")
    print(f"  Référence Monte-Carlo ({mc['n_runs']} runs)   : "
          f"{mc['mc_mean']:.4f}  [bande 90% : {mc['mc_p05']:.4f} – {mc['mc_p95']:.4f}]")
    print()
    header = f"  {'Stratégie':<24}{'moy.':>8}{'chance%':>9}{'gain%':>8}{'z':>8}{'p':>9}  verdict"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in report["backtest"]:
        signif = "DIFFÉRENT du hasard" if r["p_value"] < 0.01 else "= hasard"
        print(f"  {r['name']:<24}{r['mean_main_matches']:>8.4f}"
              f"{r['chance_hit_rate'] * 100:>8.2f}%{r['prize_rate'] * 100:>7.2f}%"
              f"{r['z_vs_null']:>8.2f}{r['p_value']:>9.3f}  {signif}")
    print()
    print("=" * 74)
    print(" CONCLUSION")
    print("-" * 74)
    beats = [r["name"] for r in report["backtest"]
             if r["p_value"] < 0.01 and r["name"] != "aleatoire (reference)"]
    if beats:
        print("  Stratégies statistiquement différentes du hasard :")
        for b in beats:
            print(f"    - {b}  (à vérifier : signal réel ou surapprentissage ?)")
    else:
        print("  Aucune stratégie ne bat le hasard de façon significative (p >= 0.01).")
        print("  C'est le résultat attendu : des tirages indépendants et")
        print("  équiprobables ne contiennent aucun motif exploitable.")
    print("  L'espérance reste négative quelle que soit la méthode : le seul")
    print("  'ultra prédictible' du Loto est la perte moyenne du joueur.")
    print("=" * 74)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Évaluation scientifique des méthodes de prédiction Loto."
    )
    parser.add_argument("--csv", default="tirages_loto.csv",
                        help="Chemin du CSV des tirages.")
    parser.add_argument("--min-history", type=int, default=200,
                        help="Nombre de tirages d'historique avant d'évaluer.")
    parser.add_argument("--mc-runs", type=int, default=300,
                        help="Nombre de simulations Monte-Carlo pour la référence.")
    parser.add_argument("--seed", type=int, default=12345, help="Graine aléatoire.")
    parser.add_argument("--json", default=None, help="Écrire le rapport JSON ici.")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        # Repli : chercher à la racine du projet si lancé depuis script/
        alt = Path(__file__).resolve().parent.parent / args.csv
        if alt.exists():
            csv_path = alt
        else:
            parser.error(f"CSV introuvable : {args.csv}")

    report = build_report(csv_path, args.min_history, args.mc_runs, args.seed)
    print_report(report)

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                   encoding="utf-8")
        print(f"\nRapport JSON écrit : {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
