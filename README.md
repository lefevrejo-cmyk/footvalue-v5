# FootValue V5

Outil éducatif d'estimation probabiliste de scores football.
Comparaison de cotes avec marge retirée. **Aucun placement de paris.**

---

## Structure du projet

```
footvalue/
├── public/
│   ├── index.html          interface
│   ├── app.js              moteur de calcul et logique UI
│   └── style.css           mise en forme
├── netlify/functions/
│   └── football.js         proxy API (la clé reste côté serveur)
├── netlify.toml            configuration Netlify
├── package.json            métadonnées du projet
└── .gitignore
```

---

## Installation

### 1. Récupère le dépôt

```bash
git clone https://github.com/<ton-compte>/footvalue.git
cd footvalue
```

Ou crée un nouveau dépôt et dépose-y ces fichiers.

### 2. Clé API

Crée un compte gratuit sur [football-data.org](https://www.football-data.org/client/register)
et récupère ton token API.

**Ne mets jamais ce token dans un fichier du dépôt.** Il va uniquement dans
les variables d'environnement Netlify.

### 3. Netlify

1. Va sur [app.netlify.com](https://app.netlify.com/) et clique sur
   *Add new site* → *Import an existing project* → GitHub.
2. Sélectionne ton dépôt.
3. Paramètres de build :
   - **Build command** : *(laisser vide)*
   - **Publish directory** : `public`
   - **Functions directory** : `netlify/functions`
4. Après le déploiement : *Site configuration* → *Environment variables* →
   *Add a variable* :

   | Clé | Valeur |
   |---|---|
   | `FOOTBALL_DATA_TOKEN` | ton token API |

5. Redéploie (*Deploys* → *Trigger deploy* → *Clear cache and deploy site*).

La clé n'est jamais servie au navigateur : la fonction l'injecte côté serveur
dans l'en-tête `X-Auth-Token`.

---

## Développement local

```bash
npm install
npx netlify dev
```

Pour tester la fonction en local, crée un fichier `.env` à la racine
(ignoré par Git) :

```
FOOTBALL_DATA_TOKEN=ton_token_ici
```

---

## Le modèle

Estimation des espérances de buts (λ) par équipe, puis conversion en
probabilités de marché.

**1. Forme pondérée par récence.**
Chaque match pèse `exp(-ξ · jours_écoulés)` avec `ξ = 0,0025`
(demi-vie ≈ 9 mois). Un match récent compte nettement plus qu'un match
de début de saison.

**2. Ajustement attaque × défense.**
L'espérance de buts d'une équipe combine son attaque et la défense de
l'adversaire, chacune rapportée à la moyenne de la compétition. Cela
corrige le fait que marquer 3 buts contre le dernier n'a pas la même
valeur que contre le leader.

**3. Shrinkage bayésien.**
`poids = n / (n + K)` avec `K = 12`. À faible échantillon, l'estimation
est ramenée vers la moyenne de la compétition ; à mesure que les matchs
s'accumulent, les données prennent le dessus. Évite le saut brutal
quand une équipe passe de 9 à 10 matchs.

**4. Correction Dixon-Coles.**
Poisson à deux λ indépendants sous-estime les scores faibles (0-0, 1-0,
0-1, 1-1), donc les matchs nuls. La correction `tau` rétablit ces scores
avec un paramètre `rho` (fixé à `-0,08` par défaut).

**5. Comparateur de cotes.**
La probabilité implicite brute `1/cote` inclut la marge du bookmaker.
FootValue la normalise sur les trois issues du 1X2 avant de comparer.
Sans cette étape, tout pari paraît favorable.

---

## Paramètres ajustables

Dans `public/app.js`, en haut du fichier :

| Constante | Rôle | Valeur par défaut |
|---|---|---|
| `XI` | vitesse de décroissance par récence | `0,0025` (demi-vie ~9 mois) |
| `SHRINK_K` | force du shrinkage bayésien | `12` |
| `RHO_DEFAULT` | paramètre Dixon-Coles | `-0,08` |
| `LEAGUE_AVG` | moyenne de buts de la compétition | `1,50 / 1,22 / 2,72` |
| `FORM_DEPTH` | matchs retenus par équipe | `10` |

`LEAGUE_AVG` est calibré pour les grands championnats européens. Pour la
Ligue 1 ou l'Eredivisie, ajuste selon les moyennes réelles de la saison.

---

## Limites

- Le modèle n'est **pas encore calibré par un backtest hors échantillon**.
- `rho` est fixé, pas estimé par maximum de vraisemblance.
- Blessures, compositions probables, fatigue et changements tactiques
  ne sont pas pris en compte.
- Le forfait gratuit de football-data.org limite les compétitions et la
  profondeur de l'historique.
- Le combiné multiplie les probabilités en supposant l'indépendance des
  sélections. Les sélections corrélées rendent le résultat trompeur.
- **Outil éducatif de simulation. Aucune connexion à Winamax, aucun
  placement de paris.**

---

## Pistes pour la suite

1. **Estimer `rho` par maximum de vraisemblance** sur l'historique de la
   compétition, plutôt que de le fixer.
2. **Alimenter `LEAGUE_AVG`** depuis les données réellement récupérées
   au lieu de constantes.
3. **Backtest hors échantillon** : garder 20 % des matchs d'une saison,
   comparer les probabilités estimées aux cotes de clôture.
4. **Combiner modèle et marché** : `p = w·p_modèle + (1−w)·p_marché`,
   le marché étant très bien calibré.
5. **Estimer l'avantage domicile par équipe** au lieu d'un facteur global.
6. **Table d'alias** pour les libellés Winamax (le modèle compare les noms
   d'équipes à l'identique, une différence de libellé casse la jointure).
