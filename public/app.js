/* ============================================================
   FootValue V5.1 — moteur de calcul
   Dixon-Coles + pondération exponentielle + shrinkage bayésien.
   Comparateur 1X2 avec marge réellement retirée.
   ============================================================ */

const $ = id => document.getElementById(id);

/* ---------- État global ---------- */
const state = {
  selected: null,      // { id, home, away, date }
  model: null,         // { lh, la, rho, home, away, fallback }
  picks: [],
  history: [],
  lastError: null
};

/* ---------- Paramètres du modèle (ajustables) ---------- */

// Décroissance exponentielle de la pondération par récence.
// xi = 0,0025 -> demi-vie ~ 277 jours (~9 mois).
const XI = 0.0025;

// Force du shrinkage bayésien : n / (n + K).
const SHRINK_K = 12;

// Correction Dixon-Coles. Négatif = plus de nuls que Poisson pur.
const RHO_DEFAULT = -0.08;

// Moyennes de buts de la compétition (ancrage du shrinkage).
const LEAGUE_AVG = { home: 1.50, away: 1.22, total: 2.72 };

// Matchs retenus par équipe.
const FORM_DEPTH = 10;

/* ---------- Utilitaires ---------- */
function status(id, msg, kind = '') {
  const el = $(id);
  if (!el) return;
  el.className = 'status ' + kind;
  el.textContent = msg;
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/* ---------- Appel à la fonction Netlify ---------- */
async function api(path) {
  const res = await fetch('/.netlify/functions/football?path=' + encodeURIComponent(path));
  if (!res.ok) {
    let msg = 'Erreur API ' + res.status;
    if (res.status === 403) msg = 'Accès refusé : vérifie la clé et la couverture gratuite de cette compétition.';
    if (res.status === 429) msg = 'Limite de requêtes atteinte. Attends avant de réessayer.';
    if (res.status === 500) msg = 'Le secret FOOTBALL_DATA_TOKEN est absent dans Netlify.';
    throw new Error(msg);
  }
  return res.json();
}

/* ============================================================
   1. MATHÉMATIQUES
   ============================================================ */

function poissonPmf(k, l) {
  let p = Math.exp(-l);
  for (let i = 1; i <= k; i++) p *= l / i;
  return p;
}

// Correction Dixon-Coles sur les scores faibles (0-0, 0-1, 1-0, 1-1).
function dcTau(x, y, lh, la, rho) {
  if (x === 0 && y === 0) return 1 - lh * la * rho;
  if (x === 0 && y === 1) return 1 + lh * rho;
  if (x === 1 && y === 0) return 1 + la * rho;
  if (x === 1 && y === 1) return 1 - rho;
  return 1;
}

// Matrice des scores -> probabilités de tous les marchés.
function calcProbs(lh, la, rho = RHO_DEFAULT) {
  let home = 0, draw = 0, away = 0, over = 0, btts = 0, total = 0;
  const MAX = 10;

  for (let h = 0; h <= MAX; h++) {
    for (let a = 0; a <= MAX; a++) {
      let p = dcTau(h, a, lh, la, rho) * poissonPmf(h, lh) * poissonPmf(a, la);
      if (p < 0) p = 0;
      total += p;
      if (h > a) home += p;
      else if (h === a) draw += p;
      else away += p;
      if (h + a >= 3) over += p;
      if (h > 0 && a > 0) btts += p;
    }
  }

  if (total > 0) { home /= total; draw /= total; away /= total; over /= total; btts /= total; }

  return {
    '1': home,
    'N': draw,
    '2': away,
    'Plus de 2,5': over,
    'Moins de 2,5': 1 - over,
    'Les deux marquent : oui': btts,
    'Les deux marquent : non': 1 - btts
  };
}

/* ============================================================
   2. ESTIMATION DES FORCES D'ÉQUIPE
   ============================================================ */

function weightedMean(values, weights) {
  const sw = weights.reduce((a, b) => a + b, 0);
  if (sw === 0) return { mean: 0, n: 0, weight: 0 };
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i] * weights[i];
  return { mean: s / sw, n: values.length, weight: sw };
}

function shrink(observed, prior, n) {
  const w = n / (n + SHRINK_K);
  return w * observed + (1 - w) * prior;
}

function teamRates(matches, teamName, now) {
  const rows = [];

  for (const m of matches) {
    const isHome = m.homeTeam?.name === teamName;
    const isAway = m.awayTeam?.name === teamName;
    if (!isHome && !isAway) continue;

    const gf = isHome ? m.score.fullTime.home : m.score.fullTime.away;
    const ga = isHome ? m.score.fullTime.away : m.score.fullTime.home;
    const days = (now - new Date(m.utcDate)) / 86400000;
    const w = Math.exp(-XI * days);

    rows.push({ gf, ga, w, isHome });
  }

  rows.sort((a, b) => b.w - a.w);
  const kept = rows.slice(0, FORM_DEPTH);

  const att = weightedMean(kept.map(r => r.gf), kept.map(r => r.w));
  const def = weightedMean(kept.map(r => r.ga), kept.map(r => r.w));

  return {
    n: kept.length,
    gf: kept.reduce((a, r) => a + r.gf, 0),
    ga: kept.reduce((a, r) => a + r.ga, 0),
    attRaw: att.mean,
    defRaw: def.mean,
    attEff: att.weight,
    defEff: def.weight
  };
}

function lambdas(home, away) {
  const priorHome = LEAGUE_AVG.home;
  const priorAway = LEAGUE_AVG.away;

  const ratio = (att, def, prior) => (att / prior) * (def / prior);

  const xgHome = ratio(home.attRaw, away.defRaw, LEAGUE_AVG.total / 2) * priorHome;
  const xgAway = ratio(away.attRaw, home.defRaw, LEAGUE_AVG.total / 2) * priorAway;

  const nHome = Math.min(home.attEff, away.defEff);
  const nAway = Math.min(away.attEff, home.defEff);

  const lh = shrink(xgHome, priorHome, Math.min(nHome, FORM_DEPTH));
  const la = shrink(xgAway, priorAway, Math.min(nAway, FORM_DEPTH));

  return {
    lh: clamp(lh, 0.25, 3.8),
    la: clamp(la, 0.20, 3.4)
  };
}

/* ============================================================
   3. CHARGEMENT ET RENDU
   ============================================================ */

async function loadMatches() {
  $('matches').innerHTML = '<p class="muted">Chargement…</p>';
  status('apiStatus', 'Connexion à l\'API…');

  try {
    const date = $('date').value;
    const comp = $('competition').value;
    const data = await api(`competitions/${comp}/matches?dateFrom=${date}&dateTo=${date}`);
    const matches = (data.matches || []).filter(m => m.homeTeam && m.awayTeam);

    if (!matches.length) {
      $('matches').innerHTML = '<p class="muted">Aucun match pour cette date et cette compétition (ou hors couverture gratuite).</p>';
      status('apiStatus', 'Requête réussie, mais aucun match disponible pour cette date.', 'ok');
      return;
    }

    $('matches').innerHTML = matches.map(m => `
      <div class="match">
        <div class="matchtop">
          <div class="teams">${esc(m.homeTeam.name)}<br>vs ${esc(m.awayTeam.name)}</div>
          <div class="score">${m.score?.fullTime?.home ?? '–'} : ${m.score?.fullTime?.away ?? '–'}</div>
        </div>
        <div class="small">${esc((m.utcDate || '').replace('T', ' ').slice(0, 16))} UTC · ${esc(m.status || '')}</div>
        <div class="actions">
          <button class="secondary" data-home="${esc(m.homeTeam.name)}" data-away="${esc(m.awayTeam.name)}" data-date="${esc(m.utcDate || '')}" data-id="${m.id}">Analyser ce match</button>
        </div>
      </div>`).join('');

    $('matches').querySelectorAll('button[data-id]').forEach(btn => {
      btn.onclick = () => chooseMatch(
        btn.dataset.id, btn.dataset.home, btn.dataset.away, btn.dataset.date
      );
    });

    status('apiStatus', matches.length + ' match(s) récupéré(s).', 'ok');
  } catch (e) {
    $('matches').innerHTML = '<p class="muted">Impossible de charger les matchs.</p>';
    status('apiStatus', e.message, 'err');
  }
}

function chooseMatch(id, home, away, date) {
  state.selected = { id, home, away, date };
  state.lastError = null;

  $('analysis').classList.remove('hidden');
  $('selectedMatch').innerHTML =
    `<div class="teams">${esc(home)} vs ${esc(away)}</div>
     <div class="small">${esc((date || '').replace('T', ' ').slice(0, 16))} UTC</div>`;
  $('probabilities').innerHTML = '<p class="muted">Récupération des derniers résultats…</p>';
  ['odd1', 'oddN', 'odd2'].forEach(f => { if ($(f)) $(f).value = ''; });
  $('analysisStatus').textContent = '';

  loadTeamStats();
}

async function loadTeamStats() {
  if (!state.selected) return;

  try {
    const comp = $('competition').value;
    const d = await api(`competitions/${comp}/matches?status=FINISHED`);
    const all = (d.matches || []).filter(m =>
      m.status === 'FINISHED' &&
      m.score?.fullTime?.home != null &&
      m.score?.fullTime?.away != null
    );

    if (all.length < 10) {
      throw new Error('Pas assez de matchs terminés dans cette compétition pour estimer la forme.');
    }

    const now = Date.now();
    const home = teamRates(all, state.selected.home, now);
    const away = teamRates(all, state.selected.away, now);

    if (home.n === 0 || away.n === 0) {
      throw new Error('Aucun match récent retrouvé pour l\'une des deux équipes (noms non concordants ou historique insuffisant).');
    }

    const { lh, la } = lambdas(home, away);
    state.model = { lh, la, rho: RHO_DEFAULT, home, away, fallback: false };
  } catch (e) {
    state.model = {
      lh: LEAGUE_AVG.home,
      la: LEAGUE_AVG.away,
      rho: RHO_DEFAULT,
      home: { n: 0, gf: 0, ga: 0 },
      away: { n: 0, gf: 0, ga: 0 },
      fallback: true
    };
    state.lastError = e.message;
  }

  renderAnalysis();
}

function renderAnalysis() {
  const m = state.model;
  if (!m) return;

  const metrics = [
    ['Matchs retenus dom.', m.home.n],
    ['Buts marqués dom.', m.home.gf],
    ['Buts encaissés dom.', m.home.ga],
    ['Matchs retenus ext.', m.away.n],
    ['Buts marqués ext.', m.away.gf],
    ['Buts encaissés ext.', m.away.ga]
  ];
  $('statsMetrics').innerHTML = metrics.map(([label, val]) =>
    `<div class="metric"><span>${esc(label)}</span><strong>${esc(val)}</strong></div>`
  ).join('');

  const probs = calcProbs(m.lh, m.la, m.rho);
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);

  $('probabilities').innerHTML = entries.map(([k, v]) => `
    <div class="pick">
      <b>${(v * 100).toFixed(1)}%</b>
      <span>${esc(k)}</span>
      <div class="bar"><div style="width:${clamp(v * 100, 0, 100)}%"></div></div>
    </div>`).join('');

  if (m.fallback) {
    status('analysisStatus',
      'Estimation générique de secours : ' + esc(state.lastError || 'données insuffisantes') +
      ' Ce n\'est pas une prédiction fondée sur les données du match.', 'warn');
  } else {
    status('analysisStatus',
      'Espérances de buts : ' + m.lh.toFixed(2) + ' (dom.) / ' + m.la.toFixed(2) + ' (ext.). ' +
      'Calculés sur ' + m.home.n + ' et ' + m.away.n + ' matchs pondérés par récence.', '');
  }

  updateValueComparator();
}

/* ============================================================
   4. COMPARATEUR 1X2 — marge réellement retirée
   ============================================================

   La marge du bookmaker porte sur l'ensemble du marché, pas sur une
   issue isolée. On la calcule donc sur les trois cotes ensemble, puis
   on la répartit proportionnellement. Seule méthode défendable.
   ============================================================ */

function removeMargin(odds) {
  const raw = odds.map(o => 1 / o);
  const sum = raw.reduce((a, b) => a + b, 0);
  return {
    overround: sum - 1,
    raw,
    fair: raw.map(p => p / sum)
  };
}

function updateValueComparator() {
  const m = state.model;
  const pick = state.picks.find(p => p.matchId === state.selected?.id) ? null : null;

  const o1 = Number($('odd1')?.value);
  const oN = Number($('oddN')?.value);
  const o2 = Number($('odd2')?.value);
  const allValid = [o1, oN, o2].every(o => Number.isFinite(o) && o > 1);

  const setGaps = (v1, vN, v2) => {
    $('gap1').textContent = v1;
    $('gapN').textContent = vN;
    $('gap2').textContent = v2;
  };

  if (!state.selected || !m || !allValid) {
    $('valueFairOdds').textContent = '—';
    $('valueOverround').textContent = '—';
    setGaps('—', '—', '—');
    $('marketTable').innerHTML = '';
    status('valueVerdict', 'Saisis les trois cotes 1, N et 2 pour lancer la comparaison.');
    return;
  }

  const probs = calcProbs(m.lh, m.la, m.rho);
  const market = removeMargin([o1, oN, o2]);

  if (market.overround <= 0) {
    status('valueVerdict', 'Ces trois cotes sont incohérentes : la somme des probabilités implicites est inférieure à 1. Vérifie ta saisie.', 'err');
    return;
  }

  const keys = ['1', 'N', '2'];
  const modelArr = keys.map(k => probs[k]);
  const edges = keys.map((k, i) => (modelArr[i] - market.fair[i]) * 100);

  // Cote théorique du modèle, affichée pour le 1 par défaut : on prend
  // l'issue où le modèle est le plus confiant, c'est le repère le plus lisible.
  const bestIdx = edges.indexOf(Math.max(...edges));
  $('valueFairOdds').textContent = (1 / modelArr[bestIdx]).toFixed(2) + ' (' + keys[bestIdx] + ')';
  $('valueOverround').textContent = (market.overround * 100).toFixed(1) + ' %';

  const fmt = e => (e > 0 ? '+' : '') + e.toFixed(1) + ' pts';
  setGaps(fmt(edges[0]), fmt(edges[1]), fmt(edges[2]));

  $('marketTable').innerHTML = renderMarketTable(probs, market);

  // Verdict global : on retient l'issue où l'écart est le plus favorable.
  const bestEdge = edges[bestIdx];
  const labels = { '1': 'victoire domicile', 'N': 'match nul', '2': 'victoire extérieur' };
  let verdict, kind = '';

  if (bestEdge > 3) {
    verdict = `Écart favorable de ${bestEdge.toFixed(1)} points sur « ${labels[keys[bestIdx]]} » (modèle ${(modelArr[bestIdx] * 100).toFixed(1)} % contre ${(market.fair[bestIdx] * 100).toFixed(1)} % de proba juste). Marge du book retirée : ${(market.overround * 100).toFixed(1)} %. À confronter à ta propre lecture du match avant d'y croire.`;
    kind = 'ok';
  } else if (bestEdge > 0) {
    verdict = `Léger avantage sur « ${labels[keys[bestIdx]]} » (+${bestEdge.toFixed(1)} pts). Trop faible pour compenser l'incertitude du modèle. Marge retirée : ${(market.overround * 100).toFixed(1)} %.`;
    kind = 'warn';
  } else {
    verdict = `Aucune issue n'est favorable. Le bookmaker est plus confiant que le modèle sur les trois. Marge retirée : ${(market.overround * 100).toFixed(1)} %.`;
    kind = 'err';
  }

  status('valueVerdict', verdict, kind);
}

/* Rendu du tableau détaillé des trois issues. */
function renderMarketTable(modelProbs, market) {
  const labels = { '1': 'Victoire domicile', 'N': 'Match nul', '2': 'Victoire extérieur' };
  const keys = ['1', 'N', '2'];

  const rows = keys.map((k, i) => {
    const modelP = modelProbs[k];
    const fairP = market.fair[i];
    const rawP = market.raw[i];
    const edge = (modelP - fairP) * 100;

    let cls = '';
    if (edge > 2) cls = 'pos';
    else if (edge < -2) cls = 'neg';

    return `
      <tr>
        <td>${labels[k]}</td>
        <td>${(modelP * 100).toFixed(1)} %</td>
        <td>${(rawP * 100).toFixed(1)} %</td>
        <td>${(fairP * 100).toFixed(1)} %</td>
        <td class="${cls}">${edge > 0 ? '+' : ''}${edge.toFixed(1)} pts</td>
      </tr>`;
  }).join('');

  return `
    <table class="cmp">
      <thead>
        <tr><th>Issue</th><th>Modèle</th><th>Brut</th><th>Juste</th><th>Écart</th></tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>`;
}

/* ============================================================
   5. COMBINÉ
   ============================================================ */

function addPick() {
  const m = state.model;
  const o1 = Number($('odd1')?.value);
  const oN = Number($('oddN')?.value);
  const o2 = Number($('odd2')?.value);

  if (!state.selected || !m) {
    status('simStatus', 'Analyse un match avant d\'ajouter une sélection.', 'err');
    return;
  }
  if (![o1, oN, o2].every(o => Number.isFinite(o) && o > 1)) {
    status('simStatus', 'Saisis les trois cotes 1, N et 2 avant d\'ajouter une sélection.', 'err');
    return;
  }
  if (state.picks.some(p => p.matchId === state.selected.id)) {
    status('simStatus', 'Ce match a déjà une sélection. Retire-la ou choisis un autre match.', 'err');
    return;
  }

  const probs = calcProbs(m.lh, m.la, m.rho);
  const market = removeMargin([o1, oN, o2]);
  const keys = ['1', 'N', '2'];

  // On retient l'issue où l'écart modèle/marché est le plus favorable.
  const edges = keys.map((k, i) => probs[k] - market.fair[i]);
  const best = edges.indexOf(Math.max(...edges));
  const bestKey = keys[best];

  state.picks.push({
    matchId: state.selected.id,
    match: state.selected.home + ' vs ' + state.selected.away,
    market: bestKey,
    odds: [o1, oN, o2][best],
    probability: probs[bestKey]
  });

  ['odd1', 'oddN', 'odd2'].forEach(f => { if ($(f)) $(f).value = ''; });
  renderCombo();
  status('simStatus',
    'Sélection ajoutée : « ' + bestKey + ' » sur ' + state.selected.home + ' vs ' + state.selected.away +
    ' à la cote ' + [o1, oN, o2][best].toFixed(2) + '.', 'ok');
}

function renderCombo() {
  if (!state.picks.length) {
    $('comboEmpty').classList.remove('hidden');
    $('combo').innerHTML = '';
    $('comboMetrics').innerHTML = '';
    return;
  }
  $('comboEmpty').classList.add('hidden');

  $('combo').innerHTML = state.picks.map((p, i) => `
    <div class="match">
      <div class="matchtop">
        <div class="teams">${esc(p.match)}</div>
        <div class="score">${p.odds.toFixed(2)}</div>
      </div>
      <div class="small">${esc(p.market)} · proba estimée ${(p.probability * 100).toFixed(1)}%</div>
      <div class="actions"><button class="danger" data-remove="${i}">Retirer</button></div>
    </div>`).join('');

  $('combo').querySelectorAll('button[data-remove]').forEach(btn => {
    btn.onclick = () => { state.picks.splice(Number(btn.dataset.remove), 1); renderCombo(); };
  });

  const totalOdds = state.picks.reduce((a, p) => a * p.odds, 1);
  const totalProb = state.picks.reduce((a, p) => a * p.probability, 1);
  const fairness = totalProb * totalOdds;

  $('comboMetrics').innerHTML = `
    <div class="metric"><span>Sélections</span><strong>${state.picks.length}</strong></div>
    <div class="metric"><span>Cote combinée</span><strong>${totalOdds.toFixed(2)}</strong></div>
    <div class="metric"><span>Probabilité combinée</span><strong>${(totalProb * 100).toFixed(1)}%</strong></div>
    <div class="metric"><span>Rapport valeur</span><strong>${fairness.toFixed(2)}</strong></div>`;

  if (fairness < 1) {
    status('simStatus', 'Combiné défavorable : la cote offerte est inférieure à la probabilité estimée. Les sélections sont multipliées en supposant leur indépendance.', 'err');
  } else {
    status('simStatus', 'Combiné potentiellement favorable. Les sélections sont multipliées en supposant leur indépendance.', 'ok');
  }
}

function simulate() {
  if (!state.picks.length) {
    status('simStatus', 'Aucune sélection dans le combiné.', 'err');
    return;
  }

  const totalOdds = state.picks.reduce((a, p) => a * p.odds, 1);
  const totalProb = state.picks.reduce((a, p) => a * p.probability, 1);

  state.history.push({
    date: new Date().toISOString().slice(0, 16).replace('T', ' '),
    matchs: state.picks.map(p => p.match + ' [' + p.market + ']').join(' | '),
    selections: state.picks.length,
    cote: totalOdds.toFixed(2),
    probabilite: (totalProb * 100).toFixed(1) + '%',
    valeur: (totalProb * totalOdds).toFixed(2)
  });

  renderHistory();
  status('simStatus', 'Simulation enregistrée virtuellement. Aucune mise réelle.', 'ok');
}

function renderHistory() {
  if (!state.history.length) {
    $('history').innerHTML = '<p class="muted">Aucune simulation dans cette session.</p>';
    return;
  }
  $('history').innerHTML = state.history.map((h, i) => `
    <div class="match">
      <div class="matchtop">
        <div class="teams">#${i + 1} · ${esc(h.selections)} sélection(s)</div>
        <div class="score">${esc(h.cote)}</div>
      </div>
      <div class="small">${esc(h.matchs)}</div>
      <div class="small">Proba ${esc(h.probabilite)} · rapport valeur ${esc(h.valeur)} · ${esc(h.date)}</div>
    </div>`).join('');
}

function exportCsv() {
  if (!state.history.length) {
    status('simStatus', 'Rien à exporter : aucune simulation enregistrée.', 'err');
    return;
  }
  const header = ['date', 'selections', 'matchs', 'cote', 'probabilite', 'rapport_valeur'];
  const rows = state.history.map(h => [
    h.date, h.selections, '"' + h.matchs.replace(/"/g, '""') + '"',
    h.cote, h.probabilite, h.valeur
  ]);
  const csv = [header.join(','), ...rows.map(r => r.join(','))].join('\n');

  const blob = new Blob(['\ufeff' + csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'footvalue-historique.csv';
  a.click();
  URL.revokeObjectURL(url);
}

/* ---------- Initialisation ---------- */
(function init() {
  const today = new Date();
  $('date').value = [
    today.getFullYear(),
    String(today.getMonth() + 1).padStart(2, '0'),
    String(today.getDate()).padStart(2, '0')
  ].join('-');

  $('loadMatches').onclick = loadMatches;
  if ($('odd1')) $('odd1').oninput = updateValueComparator;
  if ($('oddN')) $('oddN').oninput = updateValueComparator;
  if ($('odd2')) $('odd2').oninput = updateValueComparator;
  $('addPick').onclick = addPick;
  $('simulate').onclick = simulate;
  $('exportCsv').onclick = exportCsv;
  $('clearCombo').onclick = () => {
    state.picks = [];
    renderCombo();
    status('simStatus', 'Combiné vidé.', '');
  };
})();
