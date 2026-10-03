#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests du module d'évaluation scientifique (script/loto_evaluation.py).

Ces tests verrouillent les faits mathématiques du Loto français :
les cotes exactes, l'espérance négative, et le fait qu'aucune stratégie
ne doit pouvoir battre le hasard de façon reproductible.

Lancer :  python test_loto_evaluation.py
"""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "script"))

import numpy as np  # noqa: E402

import loto_evaluation as ev  # noqa: E402


class TestExactOdds(unittest.TestCase):
    def test_total_combinations(self):
        # C(49,5) * 10 = 1 906 884 * 10
        self.assertEqual(ev.total_combinations(), 19_068_840)

    def test_main_match_distribution_sums_to_one(self):
        total = sum(ev.prob_k_main_matches(k) for k in range(0, ev.MAIN_PICK + 1))
        self.assertAlmostEqual(total, 1.0, places=12)

    def test_jackpot_probability(self):
        probs = ev.rank_probabilities()
        self.assertAlmostEqual(probs["5 + Chance"], 1 / 19_068_840, places=15)

    def test_prob_any_prize_matches_sum_of_ranks(self):
        probs = ev.rank_probabilities()
        self.assertAlmostEqual(ev.prob_any_prize(), sum(probs.values()), places=15)
        # ~1 chance sur 6 de gagner un rang quelconque.
        self.assertTrue(0.16 < ev.prob_any_prize() < 0.17)

    def test_rank_probabilities_all_positive_and_below_one(self):
        for label, p in ev.rank_probabilities().items():
            self.assertGreater(p, 0.0, label)
            self.assertLess(p, 1.0, label)


class TestExpectedValue(unittest.TestCase):
    def test_rtp_below_one(self):
        result = ev.expected_value()
        # Un jeu de hasard commercial a forcément un RTP < 1 (marge maison).
        self.assertLess(result["rtp"], 1.0)
        self.assertLess(result["esperance_nette"], 0.0)

    def test_breakeven_jackpot_is_implausibly_large(self):
        # L'EV brute ne devient positive qu'au-delà d'un jackpot énorme.
        # Ce seuil doit être très grand (dizaines de millions), ce qui montre
        # qu'il ne constitue pas une stratégie exploitable (partage + variance).
        be = ev.breakeven_jackpot()
        self.assertGreater(be, 10_000_000.0)

    def test_ev_turns_positive_above_breakeven(self):
        # Vérifie la cohérence : juste au-dessus du seuil, l'EV brute > mise.
        be = ev.breakeven_jackpot()
        prizes = dict(ev.DEFAULT_PRIZES)
        prizes["5 + Chance"] = be * 1.10
        self.assertGreater(ev.expected_value(prizes)["rtp"], 1.0)


class TestNullMoments(unittest.TestCase):
    def test_null_mean_matches(self):
        # Espérance du nombre de bons numéros = 5 * 5/49.
        self.assertAlmostEqual(ev.NULL_MEAN_MATCHES, 25 / 49, places=12)

    def test_null_variance_positive(self):
        self.assertGreater(ev.NULL_VAR_MATCHES, 0.0)


class TestDataAndBacktest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv = Path(__file__).resolve().parent / "tirages_loto.csv"

    def test_load_draws_valid_ranges(self):
        if not self.csv.exists():
            self.skipTest("tirages_loto.csv absent")
        draws = ev.load_draws(self.csv)
        self.assertGreater(len(draws), 100)
        self.assertTrue((draws.main >= 1).all() and (draws.main <= ev.MAIN_MAX).all())
        self.assertTrue((draws.chance >= 1).all() and (draws.chance <= ev.CHANCE_MAX).all())

    def test_random_strategy_hovers_near_null_mean(self):
        if not self.csv.exists():
            self.skipTest("tirages_loto.csv absent")
        draws = ev.load_draws(self.csv)
        rng = np.random.default_rng(7)
        res = ev.walk_forward(draws, ev.strat_random, "aleatoire", 200, rng)
        # La moyenne empirique doit rester proche de 25/49 (tolérance large).
        self.assertAlmostEqual(res.mean_main_matches, ev.NULL_MEAN_MATCHES, delta=0.08)

    def test_no_strategy_is_wildly_significant(self):
        """Sur du bruit, aucune stratégie ne doit afficher un z-score énorme."""
        if not self.csv.exists():
            self.skipTest("tirages_loto.csv absent")
        draws = ev.load_draws(self.csv)
        rng = np.random.default_rng(1)
        for name, strat in ev.STRATEGIES.items():
            res = ev.walk_forward(draws, strat, name, 200, rng)
            self.assertLess(abs(res.z_vs_null), 4.0,
                            f"{name} s'écarte anormalement du hasard (z={res.z_vs_null:.2f})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
