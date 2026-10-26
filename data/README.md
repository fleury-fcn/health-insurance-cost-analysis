# Données

## Source

**Medical Cost Personal Datasets**, 1 338 lignes × 7 colonnes.

- Origine : Brett Lantz, *Machine Learning with R* (Packt, 2013). Données **simulées** à
  partir de statistiques démographiques du US Census Bureau.
- Diffusion Kaggle : <https://www.kaggle.com/datasets/mirichoi0218/insurance>
- Miroir utilisé par le pipeline (accès sans authentification, ce qui le rend reproductible en CI) :
  <https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv>
- Empreinte SHA-256 vérifiée à chaque exécution :
  `505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281`

## Licence

La page Kaggle publie le jeu sous **Open Database License (ODbL) v1.0** : réutilisation,
modification et redistribution autorisées, avec attribution et partage à l'identique des
bases dérivées. `data/processed/insurance_clean.csv` est une base dérivée, redistribuée
sous la même licence. Le code du dépôt est sous licence MIT.

> Vérifiez la licence sur la page Kaggle avant tout usage commercial. Ce projet est
> pédagogique.

## Dictionnaire des variables

| Variable | Type | Description |
|----------|------|-------------|
| `age` | entier | Âge de l'assuré principal (18–64 ans) |
| `sex` | catégoriel | `female` / `male` |
| `bmi` | réel | Indice de masse corporelle (kg/m²) |
| `children` | entier | Nombre d'enfants couverts (0–5) |
| `smoker` | catégoriel | Fumeur : `yes` / `no` |
| `region` | catégoriel | Région de résidence aux États-Unis : `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | réel | **Cible** : frais médicaux individuels facturés par l'assurance (USD, annuels) |

## Fichiers

- `raw/insurance.csv` : téléchargé automatiquement par `make run` (non versionné).
- `processed/insurance_clean.csv` : sortie du nettoyage (phase 2), versionnée pour la démo.
