// Netlify Function — proxy vers football-data.org
// La clé API reste dans les variables d'environnement Netlify.
// Liste blanche de chemins : seuls les matchs de compétition sont exposés.

export default async (request) => {
  const headers = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Content-Type',
    'Access-Control-Allow-Methods': 'GET, OPTIONS',
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store'
  };

  if (request.method === 'OPTIONS') {
    return new Response('', { status: 204, headers });
  }

  if (request.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), { status: 405, headers });
  }

  const token = process.env.FOOTBALL_DATA_TOKEN;
  if (!token) {
    return new Response(
      JSON.stringify({ error: 'Secret FOOTBALL_DATA_TOKEN absent dans Netlify.' }),
      { status: 500, headers }
    );
  }

  const url = new URL(request.url);
  const raw = url.searchParams.get('path') || url.searchParams.get('chemin') || '';
  const path = raw.startsWith('/') ? raw.slice(1) : raw;

  // --- Liste blanche ---------------------------------------------------
  // Autorise uniquement :
  //   competitions/<CODE>/matches            (avec query string libre)
  //   competitions/<CODE>/teams              (utile pour la V5)
  const pathOnly = path.split('?')[0];
  const parts = pathOnly.split('/');

  const allowed =
    parts.length === 3 &&
    parts[0] === 'competitions' &&
    /^[A-Z0-9_-]{2,8}$/.test(parts[1]) &&
    (parts[2] === 'matches' || parts[2] === 'teams');

  if (!allowed) {
    return new Response(
      JSON.stringify({ error: 'Chemin API non autorisé.', cheminRecu: pathOnly }),
      { status: 400, headers }
    );
  }
  // ---------------------------------------------------------------------

  try {
    const upstream = await fetch('https://api.football-data.org/v4/' + path, {
      headers: { 'X-Auth-Token': token }
    });
    return new Response(await upstream.text(), {
      status: upstream.status,
      headers
    });
  } catch {
    return new Response(
      JSON.stringify({ error: 'Impossible de joindre football-data.org.' }),
      { status: 502, headers }
    );
  }
};

export const config = { path: '/.netlify/functions/football' };
