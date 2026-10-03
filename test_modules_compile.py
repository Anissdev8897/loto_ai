#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de régression (audit C2) : tous les fichiers Python du dépôt doivent compiler.

Cinq modules avaient été corrompus par un marqueur de troncature écrit dans le code,
les rendant non importables. Ce test garantit qu'aucun fichier ne contient ce marqueur
et que tout le code se compile (py_compile), de sorte que le problème ne puisse pas
réapparaître silencieusement.

Lancer :  python test_modules_compile.py
"""

import glob
import os
import py_compile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
# Construit par morceaux pour que ce fichier de test ne contienne pas lui-même
# la chaîne recherchée (sinon il se signalerait lui-même).
TRUNCATION_MARKER = "Content truncated " + "due to size limit"


def _python_files():
    files = glob.glob(os.path.join(HERE, "*.py"))
    files += glob.glob(os.path.join(HERE, "script", "*.py"))
    return sorted(files)


class TestAllModulesCompile(unittest.TestCase):
    def test_every_module_compiles(self):
        failures = []
        for path in _python_files():
            try:
                py_compile.compile(path, doraise=True)
            except py_compile.PyCompileError as exc:
                failures.append(f"{os.path.relpath(path, HERE)} : {exc}")
        self.assertEqual(failures, [], "Fichiers non compilables :\n" + "\n".join(failures))

    def test_no_truncation_marker_in_sources(self):
        offenders = []
        for path in _python_files():
            with open(path, encoding="utf-8", errors="replace") as fh:
                if TRUNCATION_MARKER in fh.read():
                    offenders.append(os.path.relpath(path, HERE))
        self.assertEqual(
            offenders, [],
            "Marqueur de troncature trouvé dans : " + ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
