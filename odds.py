import requests
import os
import re
import unicodedata
from functools import lru_cache

API_KEY = os.getenv("ODDS_API_KEY", "174e13ddef0e3b3707f4c37a63e589e9")

STOPWORDS = {
    "fc", "cf", "sc", "ac", "cd", "club", "de", "the", "fk", "if",
    "afc", "bk", "sv", "nk", "ks", "as", "atletico", "athletic"
}


def _normalizar_nombre(nombre):
    texto = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode("ascii")
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    tokens = [token for token in texto.split() if token and token not in STOPWORDS]
    return tokens


def _puntaje_match(nombre_a, nombre_b):
    tokens_a = set(_normalizar_nombre(nombre_a))
    tokens_b = set(_normalizar_nombre(nombre_b))

    if not tokens_a or not tokens_b:
        return 0.0

    interseccion = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return interseccion / union if union else 0.0


def _extraer_cuotas_game(game):
    bookmakers = [bk for bk in (game.get("bookmakers", []) or []) if isinstance(bk, dict)]
    if not bookmakers:
        return None
    merged = {}
    bookmaker_count = 0
    for bookmaker in bookmakers:
        markets = bookmaker.get("markets", [])
        if not markets or not isinstance(markets[0], dict):
            continue
        outcomes = markets[0].get("outcomes", [])
        if not outcomes:
            continue
        bookmaker_count += 1
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            name = outcome.get("name")
            price = outcome.get("price")
            if name is not None and price is not None and name not in merged:
                merged[name] = price

    if not merged:
        return None

    return {
        "odds": merged,
        "bookmaker_count": bookmaker_count,
    }

@lru_cache(maxsize=1)
def _get_odds_catalog():
    if not API_KEY:
        return []

    url = f"https://api.the-odds-api.com/v4/sports/soccer/odds/?regions=eu&markets=h2h&apiKey={API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    return [item for item in data if isinstance(item, dict)]


def get_odds(match_home, match_away):

    try:
        res = _get_odds_catalog()
        mejor_match = None
        mejor_puntaje = 0.0

        for game in res:
            if not isinstance(game, dict):
                continue

            home = game.get("home_team")
            away = game.get("away_team")
            if not home or not away:
                continue

            puntaje_directo = _puntaje_match(match_home, home) + _puntaje_match(match_away, away)
            puntaje_cruzado = _puntaje_match(match_home, away) + _puntaje_match(match_away, home)

            if puntaje_directo >= puntaje_cruzado:
                puntaje = puntaje_directo
                orientacion = "directa"
            else:
                puntaje = puntaje_cruzado
                orientacion = "cruzada"

            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_match = (game, orientacion)

        if not mejor_match or mejor_puntaje < 1.0:
            return None

        game, orientacion = mejor_match
        odds_pack = _extraer_cuotas_game(game)
        if not odds_pack:
            return None
        odds = odds_pack["odds"]
        bookmaker_count = odds_pack["bookmaker_count"]

        home = game.get("home_team")
        away = game.get("away_team")
        if orientacion == "directa":
            cuota_local = odds.get(home, 1.9)
            cuota_visitante = odds.get(away, 2.0)
        else:
            cuota_local = odds.get(away, 1.9)
            cuota_visitante = odds.get(home, 2.0)

        return {
            "cuota_local": cuota_local,
            "cuota_visitante": cuota_visitante,
            "cuota_empate": odds.get("Draw", 3.2),
            "market_match_score": round(mejor_puntaje / 2, 3),
            "bookmaker_count": bookmaker_count,
        }

        return None

    except Exception:
        return None
