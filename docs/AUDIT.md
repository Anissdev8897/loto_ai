> **Statut des correctifs (branche `claude/intelligent-rubin-aq0jg2`)**
> Correctifs appliqués dans cette itération :
> - **C2** : les 5 fichiers tronqués recompilent (tails reconstruits/fermés proprement) + test de régression `test_modules_compile.py`.
> - **C1 / M1** : `debug` piloté par l'environnement (défaut `False`), traceback retirée des réponses API.
> - **M9** : CORS restreint à `ALLOWED_ORIGINS` (plus de `'*'` avec credentials).
> - **M7** : validation des entrées `/api/predict` (méthode en liste blanche, `combinations` entier borné → 400).
> - **H3** : extraction de probabilité corrigée (P(classe=1) au lieu de P(non tiré)).
> - **M11** : features temporelles déterministes (date du prochain tirage, plus de `datetime.now()`).
> - **M5** : plus de suppression/ré-entraînement destructif automatique sur warning de version (`AUTO_RETRAIN` opt-in).
> - **L3** : journalisation des en-têtes/corps bruts supprimée.
> - **H5 / M14 / M15** : page publique et API honnêtes (fin du « quantique »/« prédit », bandeau jeu responsable 18+/ANJ, champ `disclaimer`, `has_predictive_value=false`).
> - **L2 / L8** : `.env.example` sans IP/compte root, README corrigé sur la source de scraping.
>
> Chantiers de fond laissés à la décision du propriétaire (voir §4) : refonte ML (walk-forward/baseline branchés au produit), unification de `LotoAnalyzer`, consommation de `config.py`, intégrité des modèles (signature/retrait de git), scraping HTTPS officiel.

---

# Rapport d'audit — Projet `loto_ai`

**Auditeur :** Fable 5.1  **Date :** 2026-10-03  **Dépôt :** `/home/user/loto_ai` (HEAD)  **Service public visé :** kenopredictionia.fr
**Base :** 53 constats retenus après vérification adverse (48 CONFIRMED, 5 PLAUSIBLE). Aucun constat hors de cette base n'est ajouté.

---

## 1. Résumé exécutif

Le projet expose un service Flask qui vend à de vrais joueurs une « prédiction » de tirages de loto/keno. Or **un tirage équiprobable et i.i.d. n'est pas prédictible** (sophisme du joueur) : il n'existe, par construction, aucune fonction des tirages passés qui anticipe le prochain. **Le risque principal du projet n'est donc pas technique mais l'écart entre la promesse commerciale et la réalité statistique** — un écart que le dépôt lui-même documente (README + `script/loto_evaluation.py` concluent RTP < 1, espérance négative, aucune stratégie ne bat le hasard), tout en continuant d'afficher au public un « super calculateur quantique » et une « IA » prédictive.

À cela s'ajoutent des défauts d'ingénierie graves : un serveur de dev lancé en `debug=True` sur `0.0.0.0` (RCE via le debugger Werkzeug), le chargement non vérifié de modèles `pickle` (RCE/supply-chain), cinq modules sources corrompus et non compilables, et un pipeline ML invalidé par une fuite temporelle. La configuration centralisée (`config.py`) existe mais n'est jamais importée : les garde-fous prévus (debug off, origines restreintes) sont contournés. **Priorité : retirer la surface RCE, aligner le discours public sur les faits, et acter la non-prédictibilité avant toute autre évolution.**

---

## 2. Tableau de synthèse (sévérité × dimension — nombre de constats)

| Sévérité | Sécurité | Bugs | Archi | ML | Honnêteté | **Total** |
|---|---|---|---|---|---|---|
| **Critical** | 1 | 1 | 2 | 2 | 0 | **6** |
| **High** | 1 | 1 | 1 | 2 | 2 | **7** |
| **Medium** | 3 | 5 | 7 | 6 | 3 | **24** |
| **Low** | 6 | 4 | 2 | 2 | 2 | **16** |
| **Total** | **11** | **11** | **12** | **12** | **7** | **53** |

> Les constats détaillés ci-dessous **regroupent par cause racine** les constats décrivant le même problème sous plusieurs dimensions (p. ex. `debug=True` signalé en sécurité, bugs et archi). Chaque regroupement est présenté à sa **sévérité maximale** et mentionne les dimensions concernées. Le tableau ci-dessus, lui, reflète le décompte brut des 53 constats retenus.

---

## 3. Constats détaillés

### 3.1 — CRITIQUES

**C1. `debug=True` codé en dur sur un service public écoutant `0.0.0.0` → RCE via le debugger Werkzeug** *(sécurité + bugs + archi)*
- **Fichier :** `app.py:1087` (et `app.py:792`)
- **Impact :** `app.run(debug=True, host='0.0.0.0', port=5000)` active le debugger interactif Werkzeug : toute exception non gérée ouvre une console Python dans le navigateur (PIN dérivable/bruteforçable) → exécution de code à distance et prise de contrôle du serveur. De plus, `app.py:792` renvoie la traceback complète dans le JSON tant que `app.debug` est vrai. `config.py:45` définit pourtant `FLASK_DEBUG` (défaut `False`), valeur jamais lue par `app.py`. OWASP A05.
- **Recommandation :** `app.run(debug=config.FLASK_DEBUG, host=config.FLASK_HOST, port=config.FLASK_PORT)` ; servir derrière gunicorn/uwsgi + reverse proxy ; supprimer le champ `traceback` de la réponse API ; ne jamais exposer le debugger.

**C2. Cinq modules `.py` corrompus par un marqueur de troncature écrit dans le code → non compilables** *(archi + bugs + sécurité)*
- **Fichiers :** `script/frequency_analysis.py:341-343`, `script/loto_backtesting.py:356`, `script/loto_incremental_learning.py:343-345`, `script/loto_gui.py:322`, `script/loto_visualization.py:382`
- **Impact :** le texte littéral `(Content truncated due to size limit...)` a été écrit dans ces fichiers. `py_compile` échoue sur les cinq (`SyntaxError` : signature de `calculate_adaptive_weights` coupée ; `IndentationError` : `if` sans corps). Conséquences en prod : apprentissage incrémental, backtesting et visualisation totalement non fonctionnels ; `frequency_analysis` est importé par `app.py` (cf. M3).
- **Recommandation :** restaurer les fichiers depuis une source saine (historique antérieur ou ré-extraction) ; ajouter `python -m py_compile`/flake8 et des tests d'import en pre-commit et en CI.

**C3. Fuite temporelle : `train_test_split` avec mélange au lieu d'une validation walk-forward** *(ML)*
- **Fichiers :** `auto_trainer.py:335`, `train_local.py:291`, `script/loto_main.py:640`, `script/loto_analyzer_improved.py:825`
- **Impact :** les 4 pipelines découpent une série chronologique avec `train_test_split(..., test_size=0.2, random_state=42)` sans `shuffle=False`. Le mélange place des tirages futurs dans le train. Pire : les features sont des fenêtres glissantes de 50 tirages — deux observations voisines partagent 49/50 de leur fenêtre, donc une observation de test côtoie dans le train des fenêtres quasi identiques. L'accuracy rapportée mesure une interpolation, pas une généralisation « vers l'avant ». Seul `script/loto_evaluation.py:317` fait du walk-forward, mais il n'est pas branché au produit.
- **Recommandation :** évaluation walk-forward / `TimeSeriesSplit` (`shuffle=False`, découpe temporelle), comme le fait déjà `loto_evaluation.py:walk_forward`.

**C4. Pipeline ML sans signal apprenable ni baseline aléatoire — contredit par le propre module d'évaluation du dépôt** *(ML)*
- **Fichier :** `auto_trainer.py:291`
- **Impact :** le RandomForest est entraîné à prédire le prochain tirage (one-hot 5-parmi-49 + Chance) à partir de fréquences passées. Des tirages i.i.d. équiprobables n'offrent aucune fonction prédictive ; aucun des 4 pipelines ne compare le modèle à une baseline aléatoire. Le dépôt contient pourtant `script/loto_evaluation.py` (walk-forward + Monte-Carlo + test de significativité) qui conclut explicitement qu'aucune stratégie ne bat le hasard (p ≥ 0.01, RTP < 1, EV nette négative) — ce que `README.md:35` reconnaît, mais que `loto_page.html:262` nie publiquement.
- **Recommandation :** acter que la tâche est non-apprenable ; ne pas présenter de « prédiction » prédictive à de vrais utilisateurs. Si un modèle est conservé, l'accompagner obligatoirement d'une baseline aléatoire et du rapport `loto_evaluation.py` ; aligner le discours public sur `README.md`.

---

### 3.2 — ÉLEVÉS (High)

**H1. Désérialisation non sécurisée : `joblib.load` (pickle) sur modèles non vérifiés → RCE** *(sécurité)*
- **Fichier :** `app.py:151-167` (aussi `loto_main.py:752`, `loto_analyzer_improved.py:930`)
- **Impact :** les 4 `.joblib` sont chargés sans contrôle d'intégrité/signature, inconditionnellement au démarrage. `pickle` exécute du code arbitraire à la désérialisation : un `.joblib` altéré dans `resultats_loto/models/` donne une RCE dans le process Flask public. Surface aggravée par les ~147 Mo de modèles versionnés dans git (cf. M8) et par la ré-entraînement/suppression automatique (cf. M5). OWASP A08.
- **Recommandation :** vérifier l'intégrité avant chargement (hash SHA-256 épinglé + signature), restreindre les droits d'écriture de `resultats_loto/models/`, sortir les blobs de git, migrer vers un format sans exécution (ONNX, skops avec liste blanche).

**H2. Trois implémentations divergentes de `LotoAnalyzer` ; la variante « improved » auto-charge depuis un répertoire inexistant** *(archi ; recoupe un constat ML medium)*
- **Fichier :** `script/loto_analyzer_improved.py:169` (et `:82`, `:90`, `:209`)
- **Impact :** deux classes homonymes `LotoAnalyzer` (`loto_analyzer_improved.py:142`, `loto_main.py:197`) plus une implémentation inline dans `app.py`, avec feature-engineering et chemins divergents (fenêtre 50 vs `window_size_for_ml=5`). `improved` pointe `model_dir` sur `resultats_loto/models_loto` (inexistant) et appelle `_load_rf_models()` dès `__init__` → ré-entraînement silencieux d'un modèle différent de celui de prod (`resultats_loto/models`). Selon le point d'entrée, « la prédiction du modèle » n'est ni la même fonction de features ni le même artefact.
- **Recommandation :** unifier sur une seule classe paramétrée consommant `config.MODELS_DIR` ; échouer explicitement si le répertoire est absent au lieu de ré-entraîner en silence.

**H3. Extraction de probabilité ML dégénérée/inversée : le top-5 est tiré de P(numéro NON tiré)** *(ML)*
- **Fichier :** `app.py:452-481` (et `loto_main.py:801`)
- **Impact :** pour un seul échantillon, `predict_proba` renvoie 49 arrays `(1, n_classes)`. Le test `len(probas[i]) > 1` (nombre de **lignes** = 1) est toujours faux → la branche correcte est du code mort ; le flux prend `probas[i][0][0]` = P(classe 0) = **P(numéro non tiré)**. Les scores triés par ordre décroissant sélectionnent donc les numéros jugés les **moins** probables. Côté `loto_main.py:801`, `probas[i][1]` lève un `IndexError` masqué.
- **Recommandation :** prendre `probas[i][0][list(estimators_[i].classes_).index(1)]` (P de la classe « 1 »), gérer l'absence de la classe 1 (prob = 0), ne pas confondre nombre de lignes et nombre de classes.

**H4. Aucun backtesting valide branché au produit ; le seul backtest « ML » ne compile pas** *(ML)*
- **Fichier :** `script/loto_backtesting.py:356`
- **Impact :** le module censé rejouer l'histoire avec `LotoAnalyzer` est tronqué (`SyntaxError`) → non exécutable. Aucune évaluation walk-forward côté pipeline (cf. C3). Le seul backtest correct (`loto_evaluation.py`) n'est appelé ni par `app.py` ni par les trainers : il n'alimente aucune décision produit. Il n'existe donc, en pratique, aucune validation hors-échantillon des « prédictions » servies.
- **Recommandation :** réparer ou retirer `loto_backtesting.py` ; adopter `loto_evaluation.py` comme évaluateur de référence et exposer son verdict ; conditionner toute « méthode » à un backtest walk-forward documenté.

**H5. Page publique : marketing trompeur (« calculateur quantique », « IA ») présentant du bruit comme prédiction, sans avertissement** *(honnêteté)*
- **Fichier :** `loto_page.html:262`
- **Impact :** la page servie à la racine sur kenopredictionia.fr vend une capacité prédictive inexistante à de vrais joueurs. Aucun calcul quantique n'existe dans le code (RandomForest sklearn), et des tirages i.i.d. ne sont pas prédictibles. L'unique encart « Information » renforce la fausse promesse au lieu d'avertir. Sur un service de pseudo-jeu d'argent, cela relève de la pratique commerciale trompeuse.
- **Recommandation :** retirer toute allégation de pouvoir prédictif et le terme « quantique »/« prédit » ; remplacer l'encart par un avertissement visible (tirages équiprobables et indépendants, aucune méthode ne peut prédire ni améliorer les chances) ; renommer « Numéros Prédits » en « Combinaison générée (aucune valeur prédictive) ».

**H6. L'honnêteté scientifique du projet est réelle mais cachée du joueur** *(honnêteté)*
- **Fichier :** `script/loto_evaluation.py:25` (et `README.md:32-41`)
- **Impact :** la reconnaissance explicite de la non-prédictibilité est confinée aux surfaces développeur (README, CLI). Le commit d'honnêteté `f45e62d` n'a touché que `README.md`, `loto_evaluation.py` et `test_loto_evaluation.py` ; `loto_page.html` n'a pas été mis à jour (dernier commit = commit initial). La couche honnête n'atteint jamais l'utilisateur qui mise.
- **Recommandation :** propager la transparence du README/eval dans la page publique et la réponse API (espérance négative, « aucune méthode ne bat le hasard ») ; aligner le discours commercial sur les faits que le projet reconnaît déjà par écrit.

---

### 3.3 — MOYENS (Medium)

**M1. Fuite de traceback complète dans la réponse API** *(sécurité)* — `app.py:792`
- **Impact :** `/api/predict` renvoie la stack trace complète (chemins absolus, structure interne, versions) tant que `app.debug` est vrai — toujours le cas (cf. C1). Fuite directe dans le corps JSON, même sans atteindre la page Werkzeug. OWASP A05.
- **Reco :** message d'erreur générique + id de corrélation ; détail loggé côté serveur uniquement. Corriger aussi `debug=True`.

**M2. Aucune authentification ni rate-limiting ; `/api/predict` déclenche des calculs lourds (DoS trivial) + entrées non validées** *(sécurité + archi)* — `app.py:395` / `app.py:404`
- **Impact :** aucune route authentifiée ni limitée. `/api/predict` (public, appelé automatiquement au chargement) effectue à chaque requête un `predict_proba` + recalculs `cycle`/`fibonacci`/`frequences` (boucles `iterrows()` sur tout l'historique) sans cache → épuisement CPU/mémoire. OWASP A04/A05.
- **Reco :** rate-limiting (Flask-Limiter), éventuelle clé API, cache des résultats (invariants entre tirages), bornage du coût par requête.

**M3. Méthodes « Fréquences » et « Optimisée » en échec silencieux : import d'un module non compilable, fallback masqué vers ML** *(bugs + archi + ML ; recoupe sécurité)* — `app.py:642` / `app.py:695`
- **Impact :** les branches `method=='frequency'` et `method=='optimized'` font `from script.frequency_analysis import calculate_number_frequencies`, module qui ne compile pas (cf. C2) → `ImportError` toujours levée, captée par un `try/except` qui ne logge qu'un warning et bascule en « Machine Learning (fallback) ». Un utilisateur choisissant « Fréquences » ou la méthode phare « Optimisée » reçoit en réalité une sortie ML (elle-même dégénérée, cf. H3), sous un libellé qui ne correspond pas au calcul. La méthode « Optimisée » (argument commercial central) ne s'exécute jamais.
- **Reco :** réparer `frequency_analysis.py` ; remonter une erreur explicite (ou désactiver proprement la méthode au boot) au lieu d'un fallback silencieux mal étiqueté ; ajouter un test d'import couvrant tous les modules de méthode.

**M4. Skew entraînement/inférence sur le modèle Chance (double standardisation)** *(bugs + ML ; recoupe archi)* — `app.py:492` ; `auto_trainer.py:379` ; `train_local.py:345`
- **Impact :** à l'entraînement, `scaler_chance.fit_transform(scaler_numbers.transform(X_train))` (double standardisation imbriquée) ; à l'inférence, `scaler_chance.transform(features)` sur features **brutes**. Les features vues par le modèle Chance ne sont pas dans le même espace qu'à l'entraînement → `predict_proba` Chance faussé. Note : `loto_main.py:683` et `loto_analyzer_improved.py:863` font un simple `fit_transform` — les 4 pipelines divergent entre eux et avec l'inférence.
- **Reco :** un unique sklearn `Pipeline` de preprocessing sérialisé, partagé train/serve ; test vérifiant `transform(train) == transform(serve)` sur un même vecteur.

**M5. Heuristique de « compatibilité » destructive au boot : un warning de version sklearn supprime les modèles et ré-entraîne à chaud** *(bugs ; recoupe archi/sécurité low)* — `app.py:1008-1020` (détection `app.py:969-991`)
- **Impact :** si un warning de chargement contient `version`/`inconsistent`/`unpickle`, `models_compatible=False` → `model_file.unlink()` sur les 4 `.joblib` puis `AutoTrainer` (RandomizedSearchCV, `n_jobs=-1`) **avant** le démarrage de Flask. Une simple différence de version scikit-learn sur le VPS détruit les modèles de prod et déclenche un ré-entraînement synchrone coûteux (boot très long, voire OOM) à chaque redémarrage.
- **Reco :** ne pas supprimer sur simple warning ; épingler scikit-learn (requirements) ; charger les modèles tels quels ; ré-entraîner uniquement sur demande explicite/hors ligne.

**M6. `config.py` n'est jamais importé par le code applicatif ; constantes dupliquées, valeurs de sécurité contournées** *(bugs + archi)* — `app.py:40` ; `config.py:1`
- **Impact :** `config.py` centralise `MAX_NUMBER`, CORS, `FLASK_DEBUG`, `MODELS_DIR`… mais n'est importé que par `test_config.py`. `app.py` redéfinit tout en dur (`MAX_NUMBER:85`, `MODELS_DIR:83`, `ALLOWED_ORIGINS:40-53`, `debug=True:1087`). Double source de vérité, config « centrale » morte, divergences déjà présentes. Défaut d'architecture le plus structurant pour la maintenabilité.
- **Reco :** faire d'`app.py` et des trainers des consommateurs de `config.py` (`app.config.from_object`) ; supprimer les duplications ; à défaut, supprimer `config.py` pour ne pas entretenir l'illusion.

**M7. Paramètre `combinations`/`method` non validé : type non entier → 500+traceback ; valeur négative → liste vide renvoyée silencieusement** *(bugs)* — `app.py:404-405`
- **Impact :** `combinations` passe dans `range(min(n, 5))` sans typage/bornes. `{'combinations': 2.5}` → `TypeError` ; `{'combinations': '3'}` → `TypeError` ; ces exceptions tombent en 500 + traceback. `{'combinations': -2}` → `range(-2)` vide → tableau `combinations` **vide** avec `success:true` (réponse malformée silencieuse).
- **Reco :** `n = max(1, min(int(num_combinations), 5))` avec erreur 400 explicite ; `method` validé contre une liste blanche.

**M8. Modèles joblib ~147 Mo committés dans git (règle `.gitignore` commentée)** *(archi)* — `.gitignore:49`
- **Impact :** `# resultats_loto/models/*.joblib` → 4 blobs (~141 Mo, dont `rf_number_model.joblib` ~100 Mo) versionnés et distribués. Chargés via `joblib.load` (pickle) inconditionnellement au boot : surface de désérialisation non vérifiée (cf. H1) + dépôt obèse.
- **Reco :** décommenter la règle, `git rm --cached` des `.joblib`, distribution hors-source (artefact store) avec vérification d'intégrité.

**M9. CORS wildcard combiné à `supports_credentials=True` ; `ALLOWED_ORIGINS` défini puis ignoré** *(archi ; aussi sécurité low)* — `app.py:57`
- **Impact :** `CORS(app, origins='*', supports_credentials=True)` est une configuration contradictoire et dangereuse (expose les réponses authentifiées à toute origine). Une liste `ALLOWED_ORIGINS` curée existe (`app.py:40-53`, `config.py:54-65`) mais n'est jamais appliquée.
- **Reco :** `CORS(app, origins=config.ALLOWED_ORIGINS, supports_credentials=True)` ; ne jamais associer `'*'` et credentials (sinon `supports_credentials=False`).

**M10. Dataset `tirages_loto.csv` dupliqué à l'identique + scripts jetables + logging global à l'import** *(archi)* — `script/tirages_loto.csv:1`
- **Impact :** `tirages_loto.csv` existe en double (racine et `script/`, byte-identique), les deux suivis par git, alors que `AutoScraper` en réécrit un → deux sources de vérité. S'ajoutent du code mort (`script/fix_placement.py`, `script/extract_methods.py`) et 12 modules appelant `logging.basicConfig(FileHandler('log.txt'))` à l'import (chemin relatif au CWD), imposant la config de logging du process.
- **Reco :** un seul CSV canonique (référencé via `config.CSV_FILE`) ; retirer les scripts jetables ; configurer le logging au seul point d'entrée, chemin absolu.

**M11. Prédiction non déterministe : features temporelles basées sur `datetime.now()`** *(archi + ML + honnêteté ; aussi bugs low)* — `app.py:293-297`
- **Impact :** `prepare_features` injecte `datetime.now()` (jour, mois, semaine ISO) comme features du « prochain tirage ». À données inchangées, la « prédiction » varie selon l'heure/jour de la requête (non reproductible, non testable). Ces features correspondent à « aujourd'hui », qui n'est généralement pas un jour de tirage → distribution différente de l'entraînement (features dérivées de la vraie date du tirage). Elles encodent de toute façon du bruit pur.
- **Reco :** retirer les features temporelles (aucun pouvoir prédictif) ou figer la date cible sur le prochain jour de tirage réel ; garantir le déterminisme.

**M12. Pondération par phase lunaire comme feature prédictive** *(ML)* — `script/cycle_analysis.py:559`
- **Impact :** `analyze_lunar_cycles` surpondère pleine/nouvelle lune (`phase_importance = 1.5`). La phase lunaire n'a aucun effet causal sur un RNG/tirage de boules ; le surpoids amplifie du bruit en « signal ». Ces poids nourrissent la méthode « optimized » exposée aux utilisateurs. Cas d'école d'illusion du joueur encodée en dur.
- **Reco :** supprimer la pondération lunaire ; si une analyse descriptive est conservée, afficher un test de significativité montrant la compatibilité avec le hasard.

**M13. Méthodes hot/cold et Fibonacci inverse : data snooping sur du bruit** *(ML)* — `script/fibonacci_weighting.py:90`
- **Impact :** `apply_inverse_fibonacci_weights` classe les numéros par fréquence passée puis affecte des poids de Fibonacci selon le rang — deux croyances sans fondement sur des tirages i.i.d. Ces poids sont mélangés aux scores ML (méthodes « fibonacci » et « optimized »). `loto_evaluation.py:258-272` inclut d'ailleurs `strat_fibonacci` précisément pour montrer qu'elle ne bat pas le hasard.
- **Reco :** ne pas présenter ces pondérations comme prédictives ; les réserver à un affichage descriptif explicitement étiqueté « sans valeur prédictive », ou les retirer.

**M14. Aucune mention de jeu responsable, d'espérance négative, de restriction 18+/ANJ sur toute surface vue par l'utilisateur** *(honnêteté)* — `loto_page.html:260`
- **Impact :** aucun avertissement de jeu responsable, aucune info sur l'espérance/RTP, aucune restriction d'âge, aucune référence au régulateur (ANJ) ni lien d'aide, ni sur la page ni dans l'API. La seule ligne « Jeu Responsable » est dans `README.md:180` (développeur). Lacune de conformité majeure pour un produit relié à un public misant de l'argent.
- **Reco :** bandeau permanent (page + API) : interdit aux moins de 18 ans, jeu de hasard à espérance négative (RTP ~40 %), coordonnées d'aide (09 74 75 13 13 / joueurs-info-service) ; vérifier les obligations ANJ applicables.

**M15. Le contrat API présente les numéros comme « predictions » sans disclaimer** *(honnêteté)* — `app.py:772`
- **Impact :** le JSON de `/api/predict` renvoie une clé `prediction` avec `recommended_combination`, une carte `number_predictions` (scores type probabilité) et des labels valorisants (« Machine Learning », « Optimisée (ML + Fibonacci + Cycles + Fréquences) »). Aucun champ n'indique l'absence de valeur prédictive. La structure même du contrat institutionnalise la fausse promesse.
- **Reco :** renommer (`generated_combination` au lieu de `prediction`), re-étiqueter/supprimer `number_predictions` (ce ne sont pas des probabilités de gain), inclure un champ `disclaimer` repris à l'affichage.

---

### 3.4 — FAIBLES (Low)

**L1. Empoisonnement de données : scraping en HTTP clair d'un site tiers non officiel** *(sécurité + bugs)* — `script/loto_web_scraper.py:108`
- **Impact :** `scrap_loto_numbers()` récupère l'historique via `http://loto.akroweb.fr/` (HTTP, pas de TLS, pas de vérification de provenance). Un MITM ou une modification du site peut injecter des valeurs restant dans les plages validées (1-49 / 1-10) → écrites dans `tirages_loto.csv` puis ingérées par les modèles. Parsing fragile (`soup.find('table')`, index fixes) → DataFrame vide/lignes ignorées en silence sur changement de structure. OWASP A08/A03.
- **Reco :** source officielle (FDJ) en HTTPS avec vérification TLS ; valider la cohérence (nombre de lignes, dates attendues) ; refuser l'ingestion en cas d'anomalie.

**L2. Topologie d'infrastructure et compte root exposés en dur dans le dépôt** *(sécurité + archi)* — `config.py:48` ; `app.py:41`
- **Impact :** IP VPS `107.189.17.46`, domaine, chemin de déploiement et `VPS_USER='root'` en valeurs par défaut (`config.py:48-51`), répétés dans `app.py:41-52`/`1083-1086` et `loto_page.html` ; `.env.example` les réaffiche. Révèle la cible d'attaque et un déploiement en root. OWASP A05.
- **Reco :** externaliser via variables d'environnement/`.env` non committé ; vider `.env.example` ; ne jamais déployer en root ; purger l'historique git si nécessaire.

**L3. Journalisation de données sensibles (headers complets, corps JSON, IP)** `[PLAUSIBLE]` *(sécurité)* — `app.py:74`
- **Impact :** `before_request` journalise IP, Origin, l'ensemble des headers et le corps JSON de chaque requête. En niveau debug (actif), cela peut écrire d'éventuels `Cookie`/`Authorization` et le payload complet dans les logs/stdout → surface RGPD. OWASP A09.
- **Reco :** ne jamais logger headers/corps bruts ; masquer `Authorization`/`Cookie` ; limiter le niveau de log en prod ; protéger l'accès aux logs.

**L4. Rechargement concurrent des modèles globaux sans verrou + ré-entraînement destructif automatique** *(sécurité CONFIRMED ; scénario race condition bugs `[PLAUSIBLE]`)* — `auto_scraper.py:209-210` ; `app.py:115`
- **Impact :** le thread scheduler d'`AutoScraper` appelle `app.load_models()` qui réaffecte les globales `rf_number_model`/`scaler_*` pendant que Flask (multi-thread) sert `/api/predict`, sans verrou. En cas d'erreur de chargement, ces globales passent transitoirement à `None`. **[PLAUSIBLE]** scénario : rechargement un soir de tirage pendant une requête → lecture d'un modèle/scaler `None` ou désynchronisé → `AttributeError`/500. La suppression+ré-entraînement destructif automatique (cf. M5) est **CONFIRMED**.
- **Reco :** publier atomiquement un unique objet « models » (swap de référence) ou protéger lecture/écriture par `threading.Lock`/`RLock` ; retirer la suppression heuristique automatique.

**L5. Intégrité du dataset : une exception de lecture écrase le canon par les données scrapées brutes** `[PLAUSIBLE]` *(bugs)* — `auto_scraper.py:159`
- **Impact :** dans `scrape_and_update`, une exception de parsing du CSV existant (`auto_scraper.py:159-162`) bascule sur `df_final = df_new` (scrape potentiellement partiel) écrit via `save_to_csv` → perte de l'historique canonique. De même, la branche « tirages manquants » (`:118-138`) remplace intégralement le CSV. Scénario : changement HTML → scrape partiel → perte de l'historique, ensuite ré-ingéré par l'entraînement.
- **Reco :** backup avant écriture ; seuil minimal de lignes/cohérence ; en cas d'exception, conserver l'existant (`return False`) au lieu du fallback sur `df_new`.

**L6. Métrique trompeuse : `accuracy_score` (subset accuracy) sur une cible multi-label one-hot** *(ML)* — `auto_trainer.py:365`
- **Impact :** cible one-hot 49-dim (5 valeurs à 1), évaluée par `accuracy_score` qui exige l'égalité exacte des 49 labels. Le modèle prédit quasi tout à zéro → accuracy structurellement ~0.000, non informative. C'est pourtant l'unique chiffre de performance des 4 pipelines (`:365`, `train_local.py:330`, `loto_main.py:669`, `loto_analyzer_improved.py:853`).
- **Reco :** remplacer par un hit-rate/rang par numéro comparé à la baseline aléatoire (top-k vs 5/49 attendu), comme `loto_evaluation.py` (`mean_main_matches` vs `NULL_MEAN_MATCHES`).

**L7. Sélection de modèle avec CV elle-même fuitée + génération de combinaisons non reproductible** `[PLAUSIBLE]` *(ML)* — `auto_trainer.py:355`
- **Impact :** `RandomizedSearchCV` avec `cv=3` (KFold par défaut, sans ordre temporel) sur données déjà mélangées et fenêtres chevauchantes → sélection d'hyperparamètres estimée sur des plis partageant de l'information avec le train. Par ailleurs, malgré `random_state=42`, la génération finale des combinaisons s'appuie sur `np.random`/`random` non seedés → sortie « recommandée » non reproductible.
- **Reco :** `TimeSeriesSplit` comme `cv` ; seeder toutes les sources d'aléa (`np.random.default_rng` avec graine fixe) ; documenter la procédure pour la rendre auditablement reproductible.

**L8. README : « Scraping sécurisé depuis les résultats officiels » alors que la source est un tiers non officiel en HTTP clair** *(honnêteté)* — `README.md:120`
- **Impact :** la doc qualifie l'alimentation de « sécurisée » et « officiels », alors que la source est akroweb (tiers) en HTTP sans TLS ni vérification. Affirmation inexacte qui peut faussement rassurer un repreneur/opérateur sur l'intégrité des données servies au public.
- **Reco :** corriger la doc pour décrire la source réelle et son absence de garantie d'intégrité ; passer en HTTPS et alerter sur les anomalies avant ingestion.

> *Note transversale :* le constat « bait-and-switch silencieux » (honnêteté, `[PLAUSIBLE]`, `app.py:681`) est traité dans **M3** — au-delà du bug, la substitution silencieuse de la méthode demandée par du ML sans en informer l'utilisateur est aussi un problème de transparence (ajouter `method_requested` vs `method_served` + avertissement visible lors d'un fallback).

---

## 4. Plan de remédiation priorisé

### Quick wins (effort faible, impact fort — à traiter immédiatement)
1. **`debug=False`** (lire `config.FLASK_DEBUG`) et **supprimer le champ `traceback`** de la réponse API. *(C1, M1)*
2. **CORS** : `origins=config.ALLOWED_ORIGINS`, jamais `'*'` avec credentials. *(M9)*
3. **Validation des entrées** `/api/predict` : `method` en liste blanche, `combinations` entier borné 1..5, réponse 400 explicite. *(M7)*
4. **Sortir les `.joblib` de git** : décommenter `.gitignore:49` + `git rm --cached`. *(M8)*
5. **Figer le déterminisme** : retirer `datetime.now()` des features / fixer la date cible. *(M11)*
6. **Épingler scikit-learn** et **retirer la suppression automatique** des modèles sur warning de version. *(M5)*
7. **Logs** : ne plus journaliser headers/corps bruts ; masquer `Authorization`/`Cookie`. *(L3)*
8. **Secrets/infra** : vider `.env.example`, retirer IP/domaine/`root` des sources. *(L2)*
9. **Honnêteté immédiate** : retirer « quantique »/« prédit », ajouter disclaimer + bandeau jeu responsable/18+/ANJ sur la page et dans l'API. *(H5, M14, M15)*
10. **Dépôt** : supprimer le CSV dupliqué et les scripts jetables ; corriger le README (source scraping). *(M10, L8)*

### Chantiers de fond (structurants)
1. **Restaurer les 5 fichiers tronqués** + **CI** (`py_compile`, flake8, tests d'import) bloquant tout commit non compilable. *(C2, M3)*
2. **Durcir le runtime** : servir derrière **gunicorn/uwsgi + reverse proxy** ; **auth + rate-limiting** (Flask-Limiter) ; **cache** des résultats invariants. *(C1, M2)*
3. **Source unique de vérité** : faire d'`app.py` et des trainers des consommateurs de `config.py` ; **unifier `LotoAnalyzer`** (une classe, une fonction de features, un répertoire de modèles). *(H2, M6)*
4. **Refonte ML rigoureuse** : un **sklearn `Pipeline` sérialisé unique** (preprocessing identique train/serve, fin du double-scaling) ; **walk-forward / `TimeSeriesSplit`** ; **baseline aléatoire obligatoire** ; corriger le **mapping `predict_proba`** ; remplacer `accuracy_score` ; brancher **`loto_evaluation.py`** comme évaluateur de référence. *(C3, C4, H3, H4, M4, L6, L7)*
5. **Intégrité des modèles & supply-chain** : hash/signature avant chargement, format sans exécution (ONNX/skops), artefact store hors git, **verrou/swap atomique** au rechargement. *(H1, L4)*
6. **Pipeline de données fiable** : scraping **HTTPS officiel** + vérification d'intégrité + refus sur anomalie + **backup avant écrasement**. *(L1, L5)*
7. **Décision produit** : **acter la non-prédictibilité**, retirer les pondérations lunaire/Fibonacci/hot-cold présentées comme prédictives, aligner le discours public sur le README/eval. *(C4, M12, M13, H5, H6)*

---

## 5. Conformité & honnêteté

**Constat de fond.** Un tirage de loto/keno est **équiprobable et i.i.d.** : aucune fonction des tirages passés ne peut prédire le prochain (sophisme du joueur). Le dépôt le **reconnaît lui-même** — `script/loto_evaluation.py` (walk-forward + Monte-Carlo + test de significativité) et `README.md:32-41` concluent que **RTP < 1, l'espérance nette est négative, et aucune stratégie ne bat le hasard (p ≥ 0.01)**.

**Écart promesse/réalité.** La surface développeur est honnête ; la surface **publique** ne l'est pas. `loto_page.html` affiche « super calculateur quantique », « intelligence artificielle », « Numéros Prédits » ; l'API renvoie des `prediction`/`number_predictions` à l'allure de probabilités de gain. Or il n'y a aucun calcul quantique (RandomForest), le pipeline ML est invalidé (fuite temporelle, mapping de probabilité inversé, méthodes phares en fallback silencieux), et la sortie varie avec l'horloge. **L'utilisateur qui mise ne voit jamais la vérité que le projet a écrite.**

**Risques de conformité (produit relié à un public misant de l'argent) :**
- **Pratique commerciale trompeuse / publicité mensongère** : affirmation d'un pouvoir prédictif inexistant, réfuté par le propre code du dépôt. *(H5, M15, H6)*
- **Absence de communication responsable** : aucune mention 18+, aucune information sur l'espérance négative / RTP, aucune référence au régulateur (**ANJ**), aucun lien d'aide au jeu. *(M14)*

**Recommandations de mise en conformité :**
1. Retirer toute allégation prédictive et le vocabulaire « quantique/prédit » (page + API).
2. Afficher en permanence : **interdit aux moins de 18 ans**, **jeu de hasard à espérance négative (RTP ~40 %)**, **aucune méthode ne peut prédire ni améliorer les chances**, **aide : 09 74 75 13 13 / joueurs-info-service**.
3. Renommer le contrat API (`generated_combination`, champ `disclaimer`) et re-étiqueter `number_predictions`.
4. Propager la conclusion de `loto_evaluation.py` (EV négative, « aucune stratégie ne bat le hasard ») jusqu'à l'utilisateur.
5. Vérifier les obligations **ANJ** applicables à un site promouvant des grilles de loterie.

> **Position d'audit.** Tant que l'écart promesse/réalité n'est pas corrigé, c'est le **risque n° 1 du projet** — devant même les failles RCE, car il engage la responsabilité vis-à-vis de joueurs réels et contredit par écrit ce que le code du dépôt démontre déjà.