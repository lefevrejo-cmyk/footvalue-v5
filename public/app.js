/* ============================================================
   FootValue V5.3 — moteur de calcul
   Dixon-Coles + pondération par récence + shrinkage bayésien.
   Nouveautés V5.3 :
     - 40 derniers matchs par équipe, saison précédente incluse si l'API
       la fournit, pondération exponentielle (demi-vie ~90 jours)
     - shrinkage des paramètres attaque/défense vers la moyenne de la
       compétition : poids du prior = 10 / (10 + n)
     - probabilité finale = 0,7 × marché (marge retirée) + 0,3 × modèle
     - alerte « modèle incohérent » si le modèle contredit fortement le
       marché sur l'issue favorite
   ============================================================ */

const $ = id => document.getElementById(id);

/* ---------- État global ---------- */
const state = {
  selected: null,
  model: null,
  picks: [],
  history: [],
  lastError: null,
  cohort: null         // { leagueAvg, rho, nbMatchs } mémorisé par compétition
};

/* ---------- Paramètres ajustables ---------- */

// Décroissance exponentielle de la pondération par récence.
const HALF_LIFE_DAYS = 90;
const XI = Math.LN2 / HALF_LIFE_DAYS;   // ~0,0077 par jour

// Force du shrinkage bayésien : poids du prior = K / (K + n).
const SHRINK_K = 10;

// Matchs retenus par équipe pour l'estimation de forme.
const FORM_DEPTH = 40;

// Mélange avec le marché : p_finale = W · p_marché_juste + (1 − W) · p_modèle.
const MARKET_WEIGHT = 0.7;

// Seuil d'alerte : écart modèle − marché sur l'issue favorite (en points).
const INCOHERENCE_PTS = -15;

// Bornes de sécurité sur rho (le MLE peut partir très loin si l'échantillon
// est petit ; on borne pour éviter une correction absurde).
const RHO_MIN = -0.25;
const RHO_MAX = 0.10;

// Valeurs de repli si l'historique est trop court pour calibrer.
const FALLBACK_AVG = { home: 1.50, away: 1.22, total: 2.72 };
const FALLBACK_RHO = -0.08;

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

function dcTau(x, y, lh, la, rho) {
  if (x === 0 && y === 0) return 1 - lh * la * rho;
  if (x === 0 && y === 1) return 1 + lh * rho;
  if (x === 1 && y === 0) return 1 + la * rho;
  if (x === 1 && y === 1) return 1 - rho;
  return 1;
}

function calcProbs(lh, la, rho) {
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
   2. CALIBRATION SUR L'HISTORIQUE DE LA COMPÉTITION
   ============================================================ */

// Moyennes de buts réellement observées dans les matchs fournis.
function computeLeagueAvg(matches) {
  let hg = 0, ag = 0, n = 0;
  for (const m of matches) {
    const h = m.score?.fullTime?.home, a = m.score?.fullTime?.away;
    if (h == null || a == null) continue;
    hg += h; ag += a; n++;
  }
  if (n === 0) return { ...FALLBACK_AVG, n: 0 };
  return { home: hg / n, away: ag / n, total: (hg + ag) / n, n };
}

// Moyenne de buts par équipe, pondérée par la récence.
// Sert d'ancrage au shrinkage : la "moyenne de la compétition" n'est pas
// une constante inventée mais la moyenne réelle des matchs récents.
function recencyWeightedAverages(matches, now) {
  let sHome = 0, sAway = 0, sW = 0, n = 0;
  for (const m of matches) {
    const h = m.score?.fullTime?.home, a = m.score?.fullTime?.away;
    if (h == null || a == null) continue;
    const days = (now - new Date(m.utcDate)) / 86400000;
    const w = Math.exp(-XI * days);
    sHome += h * w; sAway += a * w; sW += w; n++;
  }
  if (sW === 0) return { home: FALLBACK_AVG.home, away: FALLBACK_AVG.away, n: 0 };
  return { home: sHome / sW, away: sAway / sW, n };
}

/* --- Estimation de rho par maximum de vraisemblance -------------------

   On cherche le rho qui maximise la log-vraisemblance des scores observés.
   Le terme tau de Dixon-Coles ne modifie que les scores 0-0, 0-1, 1-0, 1-1,
   donc la contribution au logarithme s'écrit simplement :

     LL(rho) = Σ [ n_ij · ln( tau_ij(rho) ) ]  +  constante indépendante de rho

   On peut donc ignorer les termes de Poisson et ne maximiser que la somme
   des logarithmes de tau sur les quatre scores concernés. C'est une
   fonction concave en rho, ce qui rend la recherche sûre.

   Les lambda de chaque match sont approchés par la moyenne de buts
   marqués/encaissés de chaque équipe sur l'ensemble de l'historique fourni.
   C'est une approximation (le vrai Dixon-Coles estime tout conjointement),
   mais elle capte l'essentiel : le taux de nuls réel de la compétition.
   ---------------------------------------------------------------------- */

function estimateRho(matches) {
  // 1. Force offensive/défensive brute par équipe, sur tout l'historique.
  const stats = new Map();   // nom -> { gf, ga, n }
  for (const m of matches) {
    const h = m.score?.fullTime?.home, a = m.score?.fullTime?.away;
    if (h == null || a == null) continue;
    const hn = m.homeTeam?.name, an = m.awayTeam?.name;
    if (!hn || !an) continue;

    if (!stats.has(hn)) stats.set(hn, { gf: 0, ga: 0, n: 0 });
    if (!stats.has(an)) stats.set(an, { gf: 0, ga: 0, n: 0 });
    const S = stats.get(hn); S.gf += h; S.ga += a; S.n++;
    const T = stats.get(an); T.gf += a; T.ga += h; T.n++;
  }

  const avg = computeLeagueAvg(matches);
  if (avg.n < 20) return { rho: FALLBACK_RHO, n: 0, calibrated: false };

  // 2. Comptage des quatre scores faibles, avec leurs lambda attendus.
  let n00 = 0, n01 = 0, n10 = 0, n11 = 0;
  const pairs = [];   // [{lh, la, x, y}]

  for (const m of matches) {
    const h = m.score?.fullTime?.home, a = m.score?.fullTime?.away;
    if (h == null || a == null) continue;
    const hn = m.homeTeam?.name, an = m.awayTeam?.name;
    const S = stats.get(hn), T = stats.get(an);
    if (!S || !T || S.n === 0 || T.n === 0) continue;

    // Espérance de buts approchée : attaque de l'un x défense de l'autre,
    // ramenée à la moyenne de la compétition.
    const lh = clamp((S.gf / S.n) * (T.ga / T.n) / (avg.home || 1.4), 0.15, 4.5);
    const la = clamp((T.gf / T.n) * (S.ga / S.n) / (avg.away || 1.2), 0.15, 4.5);

    if (h === 0 && a === 0) { n00++; pairs.push({ lh, la, x: 0, y: 0 }); }
    else if (h === 0 && a === 1) { n01++; pairs.push({ lh, la, x: 0, y: 1 }); }
    else if (h === 1 && a === 0) { n10++; pairs.push({ lh, la, x: 1, y: 0 }); }
    else if (h === 1 && a === 1) { n11++; pairs.push({ lh, la, x: 1, y: 1 }); }
  }

  const totalLow = n00 + n01 + n10 + n11;
  if (totalLow < 15) return { rho: FALLBACK_RHO, n: totalLow, calibrated: false };

  // 3. Recherche sur grille : tau est concave en rho, un balayage fin suffit.
  const logLik = (rho) => {
    let ll = 0;
    for (const p of pairs) {
      const tau = dcTau(p.x, p.y, p.lh, p.la, rho);
      if (tau <= 0) return -Infinity;
      ll += Math.log(tau);
    }
    return ll;
  };

  let bestRho = RHO_MIN, bestLL = -Infinity;
  for (let r = RHO_MIN; r <= RHO_MAX; r += 0.002) {
    const ll = logLik(r);
    if (ll > bestLL) { bestLL = ll; bestRho = r; }
  }

  return {
    rho: clamp(bestRho, RHO_MIN, RHO_MAX),
    n: totalLow,
    calibrated: true,
    detail: { n00, n01, n10, n11 }
  };
}

/* ============================================================
   3. ESTIMATION DES FORCES D'ÉQUIPE
   ============================================================ */

function weightedMean(values, weights) {
  const sw = weights.reduce((a, b) => a + b, 0);
  if (sw === 0) return { mean: 0, n: 0, weight: 0 };
  let s = 0;
  for (let i = 0; i < values.length; i++) s += values[i] * weights[i];
  return { mean: s / sw, n: values.length, weight: sw };
}

// Ramène un indice (1 = moyenne de la compétition) vers 1.
// Le poids du prior diminue quand le nombre de matchs augmente.
function shrinkWeight(n) {
  return SHRINK_K / (SHRINK_K + n);
}

function sameTeam(side, team) {
  if (!side) return false;
  if (team.id != null && side.id != null) return String(side.id) === String(team.id);
  return side.name === team.name;
}

// Attaque et défense d'une équipe, exprimées en indice relatif à la
// compétition : chaque but est rapporté à la moyenne attendue selon le
// lieu (domicile / extérieur), puis pondéré par la récence.
function teamRates(matches, team, now, leagueAvg) {
  const rows = [];
  for (const m of matches) {
    const isHome = sameTeam(m.homeTeam, team);
    const isAway = !isHome && sameTeam(m.awayTeam, team);
    if (!isHome && !isAway) continue;
    const t = new Date(m.utcDate).getTime();
    if (!(t <= now)) continue;
    rows.push({
      t,
      gf: isHome ? m.score.fullTime.home : m.score.fullTime.away,
      ga: isHome ? m.score.fullTime.away : m.score.fullTime.home,
      expGf: isHome ? leagueAvg.home : leagueAvg.away,
      expGa: isHome ? leagueAvg.away : leagueAvg.home,
      prev: !!m._prevSeason
    });
  }

  rows.sort((a, b) => b.t - a.t);
  const kept = rows.slice(0, FORM_DEPTH);

  let sGf = 0, sGa = 0, sExpGf = 0, sExpGa = 0;
  for (const r of kept) {
    const w = Math.exp(-XI * (now - r.t) / 86400000);
    sGf += w * r.gf; sExpGf += w * r.expGf;
    sGa += w * r.ga; sExpGa += w * r.expGa;
  }

  const n = kept.length;
  const attRaw = sExpGf > 0 ? sGf / sExpGf : 1;
  const defRaw = sExpGa > 0 ? sGa / sExpGa : 1;
  const priorW = shrinkWeight(n);

  return {
    n,
    nPrev: kept.filter(r => r.prev).length,
    gf: kept.reduce((a, r) => a + r.gf, 0),
    ga: kept.reduce((a, r) => a + r.ga, 0),
    attRaw,
    defRaw,
    priorW,
    att: (1 - priorW) * attRaw + priorW,
    def: (1 - priorW) * defRaw + priorW
  };
}

// Espérances de buts : attaque de l'un × défense de l'autre × moyenne
// de la compétition pour le lieu. Le shrinkage est déjà appliqué aux
// indices d'équipe, on ne le réapplique pas ici.
function lambdas(home, away, leagueAvg) {
  return {
    lh: clamp(home.att * away.def * leagueAvg.home, 0.25, 3.8),
    la: clamp(away.att * home.def * leagueAvg.away, 0.20, 3.4)
  };
}

/* ============================================================
   4. CHARGEMENT ET RENDU
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
          <button class="secondary" data-home="${esc(m.homeTeam.name)}" data-away="${esc(m.awayTeam.name)}" data-home-id="${esc(m.homeTeam.id ?? '')}" data-away-id="${esc(m.awayTeam.id ?? '')}" data-date="${esc(m.utcDate || '')}" data-id="${m.id}">Analyser ce match</button>
        </div>
      </div>`).join('');

    $('matches').querySelectorAll('button[data-id]').forEach(btn => {
      btn.onclick = () => chooseMatch(btn.dataset.id, btn.dataset.home, btn.dataset.away, btn.dataset.date,
        btn.dataset.homeId, btn.dataset.awayId);
    });

    status('apiStatus', matches.length + ' match(s) récupéré(s).', 'ok');
  } catch (e) {
    $('matches').innerHTML = '<p class="muted">Impossible de charger les matchs.</p>';
    status('apiStatus', e.message, 'err');
  }
}

function chooseMatch(id, home, away, date, homeId, awayId) {
  state.selected = { id, home, away, date, homeId: homeId || null, awayId: awayId || null };
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

// Cache de l'historique par compétition : évite un second appel API
// si l'utilisateur enchaîne plusieurs matchs de la même compétition.
const historyCache = new Map();

const isFinished = m =>
  m.status === 'FINISHED' &&
  m.score?.fullTime?.home != null &&
  m.score?.fullTime?.away != null;

// Saison en cours + saison précédente si l'abonnement API la couvre.
// En cas de refus (forfait gratuit, 403…), on continue avec la saison en cours.
async function getFinishedMatches(comp) {
  if (historyCache.has(comp)) return historyCache.get(comp);

  const d = await api(`competitions/${comp}/matches?status=FINISHED`);
  const current = (d.matches || []).filter(isFinished);

  const startYear = Number(
    d.filters?.season ??
    (d.matches || []).find(m => m.season?.startDate)?.season.startDate.slice(0, 4)
  );

  let previous = [];
  let prevAvailable = false;
  if (Number.isFinite(startYear) && startYear > 1900) {
    try {
      const p = await api(`competitions/${comp}/matches?season=${startYear - 1}&status=FINISHED`);
      const seen = new Set(current.map(m => m.id));
      previous = (p.matches || [])
        .filter(m => isFinished(m) && !seen.has(m.id))
        .map(m => ({ ...m, _prevSeason: true }));
      prevAvailable = previous.length > 0;
    } catch {
      // Saison précédente non accessible : on garde la saison en cours seule.
    }
  }

  const result = { matches: current.concat(previous), prevAvailable, nPrev: previous.length };
  historyCache.set(comp, result);
  return result;
}

async function loadTeamStats() {
  if (!state.selected) return;

  try {
    const comp = $('competition').value;
    const hist = await getFinishedMatches(comp);
    const all = hist.matches;

    if (all.length < 20) {
      throw new Error('Seulement ' + all.length + ' match(s) terminé(s) dans cette compétition : échantillon trop court pour calibrer le modèle.');
    }

    const now = Date.now();

    // --- Calibration sur la compétition (mise en cache par compétition) ---
    if (!state.cohort || state.cohort.comp !== comp) {
      const leagueAvg = recencyWeightedAverages(all, now);
      const rhoEst = estimateRho(all);
      state.cohort = {
        comp,
        leagueAvg: { home: leagueAvg.home, away: leagueAvg.away, total: leagueAvg.home + leagueAvg.away },
        rho: rhoEst.rho,
        rhoCalibrated: rhoEst.calibrated,
        rhoN: rhoEst.n,
        nbMatchs: all.length,
        prevAvailable: hist.prevAvailable,
        nbMatchsPrev: hist.nPrev
      };
    }
    const cohort = state.cohort;

    const sel = state.selected;
    const home = teamRates(all, { id: sel.homeId, name: sel.home }, now, cohort.leagueAvg);
    const away = teamRates(all, { id: sel.awayId, name: sel.away }, now, cohort.leagueAvg);

    if (home.n === 0 || away.n === 0) {
      throw new Error('Aucun match récent retrouvé pour l\'une des deux équipes (noms non concordants ou historique insuffisant).');
    }

    const { lh, la } = lambdas(home, away, cohort.leagueAvg);
    state.model = {
      lh, la,
      rho: cohort.rho,
      home, away,
      cohort,
      fallback: false
    };
  } catch (e) {
    state.model = {
      lh: FALLBACK_AVG.home,
      la: FALLBACK_AVG.away,
      rho: FALLBACK_RHO,
      home: { n: 0, nPrev: 0, gf: 0, ga: 0 },
      away: { n: 0, nPrev: 0, gf: 0, ga: 0 },
      cohort: null,
      fallback: true
    };
    state.lastError = e.message;
  }

  renderAnalysis();
}

function renderAnalysis() {
  const m = state.model;
  if (!m) return;

  const sel = state.selected || {};
  const nbLabel = t => t.n + (t.nPrev ? ` (dont ${t.nPrev} saison préc.)` : '');
  const metrics = [
    ['Matchs utilisés · ' + (sel.home || 'dom.'), nbLabel(m.home)],
    ['Buts marqués dom.', m.home.gf],
    ['Buts encaissés dom.', m.home.ga],
    ['Matchs utilisés · ' + (sel.away || 'ext.'), nbLabel(m.away)],
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

  // --- Garde-fou de cohérence : total de buts attendu vs compétition ---
  const expectedTotal = m.lh + m.la;
  const leagueTotal = m.cohort ? m.cohort.leagueAvg.total : FALLBACK_AVG.total;
  const ratioTotal = expectedTotal / leagueTotal;
  const incoherent = ratioTotal > 1.3 || ratioTotal < 0.7;

  let coherenceMsg = '';
  if (m.cohort) {
    coherenceMsg = `Total de buts attendu : ${expectedTotal.toFixed(2)} (moyenne de la compétition : ${leagueTotal.toFixed(2)}).`;
    if (incoherent) {
      coherenceMsg += ' Écart supérieur à 30 % : vérifie l\'échantillon, il y a peut-être une anomalie dans les données.';
    }
  }

  if (m.fallback) {
    status('analysisStatus',
      'Estimation générique de secours : ' + (state.lastError || 'données insuffisantes') +
      ' Ce n\'est pas une prédiction fondée sur les données du match.', 'warn');
  } else {
    const prevMsg = m.cohort?.prevAvailable
      ? `dont ${m.cohort.nbMatchsPrev} de la saison précédente`
      : 'saison précédente non disponible via l\'API';
    const shrinkMsg = `poids de la moyenne de la compétition : ${(m.home.priorW * 100).toFixed(0)} % (dom.) / ${(m.away.priorW * 100).toFixed(0)} % (ext.)`;
    const calib = m.cohort?.rhoCalibrated
      ? `rho estimé sur ${m.cohort.rhoN} scores faibles (${m.cohort.nbMatchs} matchs de la compétition)`
      : `rho de repli (${FALLBACK_RHO}) : historique trop court pour l'estimer`;

    status('analysisStatus',
      `Espérances : ${m.lh.toFixed(2)} (dom.) / ${m.la.toFixed(2)} (ext.). ` +
      `Matchs utilisés : ${m.home.n} pour ${sel.home} et ${m.away.n} pour ${sel.away} ` +
      `(pondération exponentielle, demi-vie ${HALF_LIFE_DAYS} jours ; ${shrinkMsg}). ` +
      `Moyenne de la compétition calculée sur ${m.cohort?.nbMatchs ?? 0} matchs (${prevMsg}), ${calib}. ` +
      coherenceMsg,
      incoherent ? 'warn' : '');
  }

  updateValueComparator();
}

/* ============================================================
   5. COMPARATEUR 1X2 — marge réellement retirée
   ============================================================ */

function removeMargin(odds) {
  const raw = odds.map(o => 1 / o);
  const sum = raw.reduce((a, b) => a + b, 0);
  return { overround: sum - 1, raw, fair: raw.map(p => p / sum) };
}

// Probabilité finale = W · marché juste + (1 − W) · modèle.
// Les écarts sont calculés sur cette probabilité finale ; l'écart brut
// du modèle sert uniquement au contrôle de cohérence.
function blendWithMarket(probs, market) {
  const keys = ['1', 'N', '2'];
  const model = keys.map(k => probs[k]);
  const final = model.map((p, i) => MARKET_WEIGHT * market.fair[i] + (1 - MARKET_WEIGHT) * p);
  const edges = final.map((p, i) => (p - market.fair[i]) * 100);
  const modelEdges = model.map((p, i) => (p - market.fair[i]) * 100);

  // Issue favorite = cote la plus basse = proba implicite la plus haute.
  const favIdx = market.raw.indexOf(Math.max(...market.raw));
  const incoherent = modelEdges[favIdx] < INCOHERENCE_PTS;

  return { keys, model, final, edges, modelEdges, favIdx, incoherent };
}

function renderMarketTable(blend, market) {
  const labels = { '1': 'Victoire domicile', 'N': 'Match nul', '2': 'Victoire extérieur' };

  const rows = blend.keys.map((k, i) => {
    const edge = blend.edges[i];
    let cls = '';
    if (!blend.incoherent && edge > 2) cls = 'pos';
    else if (edge < -2) cls = 'neg';
    return `<tr>
        <td>${labels[k]}${i === blend.favIdx ? ' ★' : ''}</td>
        <td class="ind">${(blend.model[i] * 100).toFixed(1)} %</td>
        <td>${(market.raw[i] * 100).toFixed(1)} %</td>
        <td>${(market.fair[i] * 100).toFixed(1)} %</td>
        <td><b>${(blend.final[i] * 100).toFixed(1)} %</b></td>
        <td class="${cls}">${edge > 0 ? '+' : ''}${edge.toFixed(1)} pts</td>
      </tr>`;
  }).join('');

  return `<table class="cmp">
      <thead><tr><th>Issue</th><th>Modèle brut</th><th>Brut</th><th>Juste</th><th>Finale</th><th>Écart</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="foot">Finale = ${Math.round(MARKET_WEIGHT * 100)} % proba juste du marché + ${Math.round((1 - MARKET_WEIGHT) * 100)} % modèle. Écart = finale − juste. « Modèle brut » est donné à titre indicatif. ★ = issue favorite (cote la plus basse).</p>`;
}

function updateValueComparator() {
  const m = state.model;

  const o1 = Number($('odd1')?.value);
  const oN = Number($('oddN')?.value);
  const o2 = Number($('odd2')?.value);
  const allValid = [o1, oN, o2].every(o => Number.isFinite(o) && o > 1);

  const setGaps = (v1, vN, v2) => {
    $('gap1').textContent = v1; $('gapN').textContent = vN; $('gap2').textContent = v2;
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

  // Avertissement live : une marge élevée signale souvent des cotes en direct,
  // qui ne sont pas comparables à une estimation d'avant-match.
  const margeElevee = market.overround > 0.12;

  const blend = blendWithMarket(probs, market);
  const { keys, final, edges } = blend;

  const bestIdx = edges.indexOf(Math.max(...edges));
  $('valueFairOdds').textContent = (1 / final[bestIdx]).toFixed(2) + ' (' + keys[bestIdx] + ')';
  $('valueOverround').textContent = (market.overround * 100).toFixed(1) + ' %';

  const fmt = e => (e > 0 ? '+' : '') + e.toFixed(1) + ' pts';
  setGaps(fmt(edges[0]), fmt(edges[1]), fmt(edges[2]));

  $('marketTable').innerHTML = renderMarketTable(blend, market);

  const bestEdge = edges[bestIdx];
  const labels = { '1': 'victoire domicile', 'N': 'match nul', '2': 'victoire extérieur' };
  const margeTxt = `Marge retirée : ${(market.overround * 100).toFixed(1)} %.`;
  let verdict, kind = '';

  if (blend.incoherent) {
    verdict = `Modèle incohérent sur ce match, écarts non fiables. ` +
      `Sur l'issue favorite (« ${labels[keys[blend.favIdx]]} »), le modèle brut est à ` +
      `${blend.modelEdges[blend.favIdx].toFixed(1)} points de la proba juste du marché.` +
      (margeElevee ? ` Marge de ${(market.overround * 100).toFixed(1)} % : cotes probablement en direct.` : '');
    kind = 'warn';
  } else if (margeElevee) {
    verdict = `Marge de ${(market.overround * 100).toFixed(1)} % : ces cotes ressemblent à du jeu en direct. ` +
      `Un modèle d'avant-match n'est pas comparable à des cotes live — le score et la minute changent tout.`;
    kind = 'warn';
  } else if (bestEdge > 3) {
    verdict = `Écart favorable de ${bestEdge.toFixed(1)} points sur « ${labels[keys[bestIdx]]} » ` +
      `(proba finale ${(final[bestIdx] * 100).toFixed(1)} % contre ${(market.fair[bestIdx] * 100).toFixed(1)} % de proba juste ; ` +
      `modèle brut ${(blend.model[bestIdx] * 100).toFixed(1)} %). ` +
      `${margeTxt} À confronter à ta propre lecture du match avant d'y croire.`;
    kind = 'ok';
  } else if (bestEdge > 0) {
    verdict = `Léger avantage sur « ${labels[keys[bestIdx]]} » (+${bestEdge.toFixed(1)} pts sur la proba finale). ` +
      `Trop faible pour compenser l'incertitude du modèle. ${margeTxt}`;
    kind = 'warn';
  } else {
    verdict = `Aucune issue n'est favorable. Le bookmaker est plus confiant que le modèle sur les trois. ${margeTxt}`;
    kind = 'err';
  }

  status('valueVerdict', verdict, kind);
}

/* ============================================================
   6. COMBINÉ
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
  const { keys, final, edges } = blendWithMarket(probs, market);
  const best = edges.indexOf(Math.max(...edges));
  const bestKey = keys[best];
  const oddsArr = [o1, oN, o2];

  state.picks.push({
    matchId: state.selected.id,
    match: state.selected.home + ' vs ' + state.selected.away,
    market: bestKey,
    odds: oddsArr[best],
    probability: final[best]
  });

  ['odd1', 'oddN', 'odd2'].forEach(f => { if ($(f)) $(f).value = ''; });
  renderCombo();
  status('simStatus',
    'Sélection ajoutée : « ' + bestKey + ' » sur ' + state.selected.home + ' vs ' + state.selected.away +
    ' à la cote ' + oddsArr[best].toFixed(2) + '.', 'ok');
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

  status('simStatus', fairness < 1
    ? 'Combiné défavorable : la cote offerte est inférieure à la probabilité estimée. Les sélections sont multipliées en supposant leur indépendance.'
    : 'Combiné potentiellement favorable. Les sélections sont multipliées en supposant leur indépendance.',
    fairness < 1 ? 'err' : 'ok');
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
  ['odd1', 'oddN', 'odd2'].forEach(f => { if ($(f)) $(f).oninput = updateValueComparator; });
  $('addPick').onclick = addPick;
  $('simulate').onclick = simulate;
  $('exportCsv').onclick = exportCsv;
  $('clearCombo').onclick = () => {
    state.picks = [];
    renderCombo();
    status('simStatus', 'Combiné vidé.', '');
  };
})();
