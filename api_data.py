import os
import re
import unicodedata
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from zoneinfo import ZoneInfo
import random
import json

import pandas as pd
import requests
from odds import get_odds

API_KEY = os.getenv("API_SPORTS_KEY", "0dddf668bc3e722a3afda4592a179671")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "matches_cache.csv")

headers = {
    "x-apisports-key": API_KEY
}

TIMEZONE = ZoneInfo("America/Bogota")
ESTADOS_EN_CURSO = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}
ESTADOS_FINALIZADOS = {"FT", "AET", "PEN", "CANC", "ABD", "AWD", "WO"}
FAST_LIMIT = 80
MAX_FIXTURES_CONTEXTO = 10
MAX_FIXTURES_ODDS_API = 16
LIGAS_PRIORITARIAS = [
    {"label": "Champions League", "keywords": ["uefa champions league"], "countries": ["world", ""], "priority": 100},
    {"label": "Europa League", "keywords": ["uefa europa league"], "countries": ["world", ""], "priority": 98},
    {"label": "Conference League", "keywords": ["uefa europa conference league", "uefa conference league"], "countries": ["world", ""], "priority": 97},
    {"label": "Copa Libertadores", "keywords": ["copa libertadores", "conmebol libertadores"], "countries": ["world", "south america", ""], "priority": 96},
    {"label": "Copa Sudamericana", "keywords": ["copa sudamericana", "conmebol sudamericana"], "countries": ["world", "south america", ""], "priority": 94},
    {"label": "Premier League", "keywords": ["premier league"], "countries": ["england"], "priority": 95},
    {"label": "La Liga", "keywords": ["la liga", "laliga"], "countries": ["spain"], "priority": 93},
    {"label": "Serie A", "keywords": ["serie a"], "countries": ["italy"], "priority": 91},
    {"label": "Bundesliga", "keywords": ["bundesliga"], "countries": ["germany"], "priority": 89},
    {"label": "Ligue 1", "keywords": ["ligue 1"], "countries": ["france"], "priority": 87},
    {"label": "Primera A Colombia", "keywords": ["primera a", "liga betplay"], "countries": ["colombia"], "priority": 86},
    {"label": "FA Cup / Copas top", "keywords": ["fa cup", "copa del rey", "coppa italia", "dfb pokal"], "countries": [], "priority": 84},
    {"label": "Championship", "keywords": ["championship"], "countries": ["england"], "priority": 80},
    {"label": "Eredivisie", "keywords": ["eredivisie"], "countries": ["netherlands"], "priority": 79},
    {"label": "Primeira Liga", "keywords": ["primeira liga", "liga portugal"], "countries": ["portugal"], "priority": 78},
    {"label": "Liga MX", "keywords": ["liga mx"], "countries": ["mexico"], "priority": 76},
    {"label": "MLS", "keywords": ["major league soccer", "mls"], "countries": ["usa", "united states", "canada"], "priority": 75},
    {"label": "Primera Argentina", "keywords": ["primera division", "liga profesional", "superliga"], "countries": ["argentina"], "priority": 74},
    {"label": "Serie A Brasil", "keywords": ["serie a"], "countries": ["brazil"], "priority": 72},
    {"label": "Copa do Brasil", "keywords": ["copa do brasil"], "countries": ["brazil"], "priority": 71},
    {"label": "J1 League", "keywords": ["j1 league"], "countries": ["japan"], "priority": 70},
    {"label": "Saudi Pro League", "keywords": ["saudi pro league", "pro league"], "countries": ["saudi arabia"], "priority": 69},
    {"label": "AFC Champions League", "keywords": ["afc champions league"], "countries": ["world", "asia", ""], "priority": 68},
    {"label": "CAF Champions League", "keywords": ["caf champions league"], "countries": ["world", "africa", ""], "priority": 67},
    {"label": "Egyptian Premier League", "keywords": ["premier league"], "countries": ["egypt"], "priority": 66},
    {"label": "South Africa PSL", "keywords": ["premier soccer league", "premiership"], "countries": ["south africa"], "priority": 65},
    {"label": "A-League Men", "keywords": ["a-league"], "countries": ["australia"], "priority": 64},
]
LIGAS_SECUNDARIAS = [
    "europa league", "conference league", "mls", "eredivisie", "primeira liga",
    "libertadores", "sudamericana", "j1 league", "saudi pro league",
    "afc champions league", "caf champions league", "premier soccer league", "a-league"
]
ESPN_LEAGUES = [
    {"code": "uefa.champions", "label": "Champions League", "country": "World", "group": "Champions League", "priority": 100},
    {"code": "uefa.europa", "label": "Europa League", "country": "World", "group": "Europa League", "priority": 98},
    {"code": "uefa.europa.conf", "label": "Conference League", "country": "World", "group": "Conference League", "priority": 97},
    {"code": "conmebol.libertadores", "label": "Copa Libertadores", "country": "South America", "group": "Copa Libertadores", "priority": 96},
    {"code": "conmebol.sudamericana", "label": "Copa Sudamericana", "country": "South America", "group": "Copa Sudamericana", "priority": 94},
    {"code": "eng.1", "label": "Premier League", "country": "England", "group": "Premier League", "priority": 95},
    {"code": "esp.1", "label": "La Liga", "country": "Spain", "group": "La Liga", "priority": 93},
    {"code": "ita.1", "label": "Serie A", "country": "Italy", "group": "Serie A", "priority": 91},
    {"code": "ger.1", "label": "Bundesliga", "country": "Germany", "group": "Bundesliga", "priority": 89},
    {"code": "fra.1", "label": "Ligue 1", "country": "France", "group": "Ligue 1", "priority": 87},
    {"code": "col.1", "label": "Primera A Colombia", "country": "Colombia", "group": "Primera A Colombia", "priority": 86},
    {"code": "eng.2", "label": "Championship", "country": "England", "group": "Championship", "priority": 80},
    {"code": "ned.1", "label": "Eredivisie", "country": "Netherlands", "group": "Eredivisie", "priority": 79},
    {"code": "por.1", "label": "Primeira Liga", "country": "Portugal", "group": "Primeira Liga", "priority": 78},
    {"code": "mex.1", "label": "Liga MX", "country": "Mexico", "group": "Liga MX", "priority": 76},
    {"code": "usa.1", "label": "MLS", "country": "United States", "group": "MLS", "priority": 75},
    {"code": "arg.1", "label": "Primera Argentina", "country": "Argentina", "group": "Primera Argentina", "priority": 74},
    {"code": "bra.1", "label": "Serie A Brasil", "country": "Brazil", "group": "Serie A Brasil", "priority": 72},
    {"code": "jpn.1", "label": "J1 League", "country": "Japan", "group": "J1 League", "priority": 70},
    {"code": "ksa.1", "label": "Saudi Pro League", "country": "Saudi Arabia", "group": "Saudi Pro League", "priority": 69},
    {"code": "afc.champions", "label": "AFC Champions League", "country": "Asia", "group": "AFC Champions League", "priority": 68},
    {"code": "caf.champions", "label": "CAF Champions League", "country": "Africa", "group": "CAF Champions League", "priority": 67},
    {"code": "egy.1", "label": "Egyptian Premier League", "country": "Egypt", "group": "Egyptian Premier League", "priority": 66},
    {"code": "rsa.1", "label": "South Africa PSL", "country": "South Africa", "group": "South Africa PSL", "priority": 65},
    {"code": "aus.1", "label": "A-League Men", "country": "Australia", "group": "A-League Men", "priority": 64},
]
LAST_API_STATUS = {
    "source": "api_football",
    "ok": True,
    "message": "",
    "details": "",
    "used_cache": False,
}


def _set_last_api_status(ok, message="", details="", used_cache=False, source="api_football"):
    LAST_API_STATUS.update({
        "source": source,
        "ok": bool(ok),
        "message": message,
        "details": details,
        "used_cache": used_cache,
    })


def get_last_api_status():
    return LAST_API_STATUS.copy()


def _guardar_cache_partidos(df):
    if df is None or df.empty:
        return
    try:
        df.to_csv(CACHE_PATH, index=False)
    except Exception:
        pass


def _cargar_cache_partidos():
    if not os.path.exists(CACHE_PATH):
        return pd.DataFrame()
    try:
        df = pd.read_csv(CACHE_PATH)
        if df.empty:
            return df
        for col in [
            "fecha_partido", "hora_partido", "estado_partido", "liga", "pais_liga",
            "grupo_liga", "local", "visitante", "fuente_cuotas", "logo_local", "logo_visitante",
            "minuto_partido"
        ]:
            if col in df.columns:
                df[col] = df[col].apply(_texto_seguro)
        if "prioridad_liga" in df.columns:
            df["prioridad_liga"] = pd.to_numeric(df["prioridad_liga"], errors="coerce").fillna(0)
        for col in ["market_match_score", "bookmaker_count", "cuota_local", "cuota_empate", "cuota_visitante"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["fuente_partidos"] = "cache_local"
        return df
    except Exception:
        return pd.DataFrame()


def _american_to_decimal(odds_value):
    if odds_value in (None, ""):
        return None
    try:
        odds_int = int(str(odds_value).replace("+", "").strip())
    except Exception:
        return None

    if odds_int > 0:
        return round((odds_int / 100) + 1, 2)
    if odds_int < 0:
        return round((100 / abs(odds_int)) + 1, 2)
    return None


def _normalizar_equipo(nombre):
    return "".join(ch.lower() for ch in str(nombre or "") if ch.isalnum())


STOPWORDS_EQUIPO = {
    "fc", "sc", "ac", "cf", "cd", "club", "de", "the", "fk", "if",
    "afc", "bk", "sv", "nk", "ks", "as", "atletico", "athletic",
}


def _tokens_equipo(nombre):
    texto = unicodedata.normalize("NFKD", str(nombre or "")).encode("ascii", "ignore").decode("ascii")
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return [token for token in texto.split() if token and token not in STOPWORDS_EQUIPO]


def _puntaje_match_equipo(nombre_a, nombre_b):
    tokens_a = set(_tokens_equipo(nombre_a))
    tokens_b = set(_tokens_equipo(nombre_b))
    if not tokens_a or not tokens_b:
        return 0.0
    interseccion = len(tokens_a & tokens_b)
    base = min(len(tokens_a), len(tokens_b))
    return interseccion / base if base else 0.0


def _to_float_safe(valor, default=None):
    try:
        return float(valor)
    except Exception:
        return default


def _extraer_cuotas_espn(competition):
    odds = competition.get("odds") or []
    if not odds:
        return None

    moneyline = odds[0].get("moneyline") if isinstance(odds[0], dict) else None
    if not moneyline:
        return None

    home = _american_to_decimal((((moneyline.get("home") or {}).get("current") or {}).get("odds")))
    draw = _american_to_decimal((((moneyline.get("draw") or {}).get("current") or {}).get("odds")))
    away = _american_to_decimal((((moneyline.get("away") or {}).get("current") or {}).get("odds")))

    if not home or not away:
        return None

    return {
        "cuota_local": home,
        "cuota_empate": draw or 3.2,
        "cuota_visitante": away,
    }


def _enriquecer_odds_desde_espn(df_partidos, fecha=None):
    if df_partidos is None or df_partidos.empty or "fuente_cuotas" not in df_partidos.columns:
        return df_partidos

    faltantes = df_partidos["fuente_cuotas"].fillna("").eq("fallback")
    if not faltantes.any():
        return df_partidos

    try:
        df_espn, _ = _fetch_matches_espn(fecha=fecha, limit=max(len(df_partidos) * 4, FAST_LIMIT))
    except Exception:
        return df_partidos

    if df_espn.empty:
        return df_partidos

    candidatos = []
    for _, row in df_espn.iterrows():
        if str(row.get("fuente_cuotas") or "") != "espn_odds":
            continue
        candidatos.append({
            "fecha_partido": str(row.get("fecha_partido") or ""),
            "liga": str(row.get("liga") or ""),
            "local": str(row.get("local") or ""),
            "visitante": str(row.get("visitante") or ""),
            "hora_partido": str(row.get("hora_partido") or ""),
            "cuota_local": row.get("cuota_local"),
            "cuota_empate": row.get("cuota_empate"),
            "cuota_visitante": row.get("cuota_visitante"),
        })

    if not candidatos:
        return df_partidos

    df_result = df_partidos.copy()

    for idx in df_result.index[df_result["fuente_cuotas"].fillna("").eq("fallback")]:
        row = df_result.loc[idx]
        fecha_row = str(row.get("fecha_partido") or "")
        liga_row = str(row.get("liga") or "")
        local_row = str(row.get("local") or "")
        visitante_row = str(row.get("visitante") or "")
        hora_row = str(row.get("hora_partido") or "")

        mejor = None
        mejor_puntaje = 0.0

        for cand in candidatos:
            if cand["fecha_partido"] != fecha_row:
                continue

            puntaje = _puntaje_match_equipo(local_row, cand["local"]) + _puntaje_match_equipo(visitante_row, cand["visitante"])

            if puntaje < 1.2:
                continue

            if liga_row and cand["liga"] and str(liga_row).lower() == str(cand["liga"]).lower():
                puntaje += 0.2

            if hora_row and cand["hora_partido"] and hora_row == cand["hora_partido"]:
                puntaje += 0.1

            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor = cand

        if not mejor:
            continue

        df_result.at[idx, "cuota_local"] = mejor["cuota_local"]
        df_result.at[idx, "cuota_empate"] = mejor["cuota_empate"]
        df_result.at[idx, "cuota_visitante"] = mejor["cuota_visitante"]
        df_result.at[idx, "fuente_cuotas"] = "espn_odds"
        df_result.at[idx, "market_match_score"] = round(min(mejor_puntaje / 2, 1.0), 3)
        df_result.at[idx, "bookmaker_count"] = 1

    return df_result


def _extraer_cuotas_api_football_item(item, live=False):
    bookmakers = item.get("bookmakers") or []
    if not bookmakers:
        return None

    preferred_names = ["1x2 (1st Half)", "1x2 - 50 minutes", "1x2"] if live else ["Match Winner", "1x2"]

    for bookmaker in bookmakers:
        bets = bookmaker.get("bets") or []
        for bet in bets:
            bet_name = str(bet.get("name") or "")
            if bet_name not in preferred_names:
                continue
            valores = bet.get("values") or []
            home = draw = away = None
            for valor in valores:
                name = str(valor.get("value") or "").lower()
                odd = _to_float_safe(valor.get("odd"))
                if odd is None:
                    continue
                if name in {"home", "1"}:
                    home = odd
                elif name in {"draw", "x"}:
                    draw = odd
                elif name in {"away", "2"}:
                    away = odd
            if home and away and draw:
                return {
                    "cuota_local": round(home, 3),
                    "cuota_empate": round(draw, 3),
                    "cuota_visitante": round(away, 3),
                }
    return None


def _puntos_forma_espn(form):
    mapping = {"W": 3, "D": 1, "L": 0}
    if not form:
        return 7
    return sum(mapping.get(ch.upper(), 0) for ch in str(form))


def _record_ppm_espn(records):
    try:
        summary = (records or [{}])[0].get("summary", "")
        wins, draws, losses = [int(x) for x in summary.split("-")[:3]]
        matches = wins + draws + losses
        if matches <= 0:
            return 1.3
        return ((wins * 3) + draws) / matches
    except Exception:
        return 1.3


def _record_summary_espn(records):
    try:
        return (records or [{}])[0].get("summary", "")
    except Exception:
        return ""


def _extraer_goleador_espn(competitor):
    leaders = competitor.get("leaders") or []
    for leader_group in leaders:
        if not isinstance(leader_group, dict):
            continue
        nombre = str(leader_group.get("name") or "").lower()
        if nombre not in {"goals", "goalsleaders"}:
            continue
        leader_items = leader_group.get("leaders") or []
        if not leader_items:
            continue
        leader = leader_items[0]
        athlete = leader.get("athlete") or {}
        return {
            "nombre": athlete.get("displayName") or athlete.get("shortName") or "",
            "goles": int(float(leader.get("value") or 0)),
        }
    return {"nombre": "", "goles": 0}


def _map_espn_status(event):
    status_type = (((event.get("status") or {}).get("type")) or {})
    if status_type.get("completed"):
        return "FT"

    state = status_type.get("state")
    detail = _texto_seguro(status_type.get("shortDetail")).lower()
    name = _texto_seguro(status_type.get("name")).lower()

    if "half" in name or detail == "ht":
        return "HT"
    if state == "in":
        return "LIVE"
    if state == "pre":
        return "NS"
    return _texto_seguro(status_type.get("shortDetail"), "NS")


def _minuto_espn(event):
    status_type = (((event.get("status") or {}).get("type")) or {})
    detail = _texto_seguro(status_type.get("shortDetail"))
    state = _texto_seguro(status_type.get("state")).lower()
    name = _texto_seguro(status_type.get("name")).lower()
    if state != "in" and "half" not in name and detail.upper() not in {"HT", "1H", "2H"}:
        return ""
    if not detail:
        return ""
    return detail


def _minuto_api_football(match):
    status = ((match.get("fixture") or {}).get("status") or {})
    short = _texto_seguro(status.get("short")).upper()
    elapsed = status.get("elapsed")
    if short == "HT":
        return "45'"
    if short in ESTADOS_EN_CURSO and elapsed not in (None, ""):
        try:
            return f"{int(elapsed)}'"
        except Exception:
            return _texto_seguro(elapsed)
    return ""


def _estimar_metricas_espn(home_comp, away_comp):
    ppm_home = _record_ppm_espn(home_comp.get("records"))
    ppm_away = _record_ppm_espn(away_comp.get("records"))
    form_home = _puntos_forma_espn(home_comp.get("form"))
    form_away = _puntos_forma_espn(away_comp.get("form"))

    strength_home = (ppm_home / 3) + (form_home / 15)
    strength_away = (ppm_away / 3) + (form_away / 15)

    goles_local = max(1.15 + (strength_home - strength_away) * 0.55 + 0.18, 0.45)
    goles_visitante = max(1.02 + (strength_away - strength_home) * 0.5, 0.35)

    return (
        round(goles_local, 2),
        round(goles_visitante, 2),
        form_home,
        form_away,
    )


def _debe_incluir_rango_48h(match_dt, estado, now_local):
    if match_dt is None:
        return False
    if estado in ESTADOS_FINALIZADOS:
        return False
    if estado in ESTADOS_EN_CURSO or estado == "LIVE":
        return match_dt.date() == now_local.date()
    return now_local <= match_dt <= (now_local + pd.Timedelta(hours=48))


@lru_cache(maxsize=128)
def _fetch_espn_scoreboard_json(url):
    response = requests.get(url, timeout=4)
    return response.json()


def _fetch_matches_espn(fecha=None, limit=20):
    now_local = datetime.now(TIMEZONE)
    fechas_consulta = [fecha] if fecha else [
        now_local.strftime("%Y%m%d"),
        (now_local + pd.Timedelta(days=1)).strftime("%Y%m%d"),
    ]

    partidos = []
    errors = []

    consultas = [
        (
            liga,
            fecha_code,
            f"https://site.api.espn.com/apis/site/v2/sports/soccer/{liga['code']}/scoreboard?dates={fecha_code}",
        )
        for liga in ESPN_LEAGUES
        for fecha_code in fechas_consulta
    ]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futuros = {
            executor.submit(_fetch_espn_scoreboard_json, url): (liga, fecha_code)
            for liga, fecha_code, url in consultas
        }
        for futuro in as_completed(futuros):
            liga, fecha_code = futuros[futuro]
            try:
                data = futuro.result()
                events = data.get("events") or []
            except Exception as exc:
                errors.append(f"{liga['code']}:{fecha_code}:{exc}")
                continue

            for event in events:
                competition = (event.get("competitions") or [{}])[0]
                competitors = competition.get("competitors") or []
                home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                if not home or not away:
                    continue

                fecha_match = _parse_fecha_partido(event.get("date"))
                estado = _map_espn_status(event)

                if fecha is None and not _debe_incluir_rango_48h(fecha_match, estado, now_local):
                    continue

                local = ((home.get("team") or {}).get("displayName")) or ((home.get("team") or {}).get("name"))
                visitante = ((away.get("team") or {}).get("displayName")) or ((away.get("team") or {}).get("name"))
                if not local or not visitante:
                    continue

                goles_local, goles_visitante, forma_local, forma_visitante = _estimar_metricas_espn(home, away)
                ppm_local = round(_record_ppm_espn(home.get("records")), 2)
                ppm_visitante = round(_record_ppm_espn(away.get("records")), 2)
                record_local = _record_summary_espn(home.get("records"))
                record_visitante = _record_summary_espn(away.get("records"))
                goleador_local = _extraer_goleador_espn(home)
                goleador_visitante = _extraer_goleador_espn(away)
                odds = _extraer_cuotas_espn(competition)

                partidos.append({
                    "fixture_id": event.get("id"),
                    "fecha_partido": fecha_match.strftime("%Y-%m-%d") if fecha_match else "",
                    "hora_partido": fecha_match.strftime("%Y-%m-%d %H:%M") if fecha_match else "",
                    "estado_partido": estado,
                    "liga": liga["label"],
                    "pais_liga": liga["country"],
                    "grupo_liga": liga["group"],
                    "prioridad_liga": liga["priority"],
                    "local": local,
                    "visitante": visitante,
                    "logo_local": (home.get("team") or {}).get("logo"),
                    "logo_visitante": (away.get("team") or {}).get("logo"),
                    "marcador_local": int(_texto_seguro(home.get("score") or 0, "0")),
                    "marcador_visitante": int(_texto_seguro(away.get("score") or 0, "0")),
                    "minuto_partido": _minuto_espn(event),
                    "goles_local": goles_local,
                    "goles_visitante": goles_visitante,
                    "forma_local": forma_local,
                    "forma_visitante": forma_visitante,
                    "racha_local": home.get("form"),
                    "racha_visitante": away.get("form"),
                    "ppm_local": ppm_local,
                    "ppm_visitante": ppm_visitante,
                    "record_local": record_local,
                    "record_visitante": record_visitante,
                    "goleador_local": goleador_local.get("nombre"),
                    "goles_goleador_local": goleador_local.get("goles"),
                    "goleador_visitante": goleador_visitante.get("nombre"),
                    "goles_goleador_visitante": goleador_visitante.get("goles"),
                    "localia": 1,
                    "cuota_local": (odds or {}).get("cuota_local", 2.2),
                    "cuota_empate": (odds or {}).get("cuota_empate", 3.2),
                    "cuota_visitante": (odds or {}).get("cuota_visitante", 2.8),
                    "fuente_cuotas": "espn_odds" if odds else "fallback",
                    "market_match_score": 0.78 if odds else 0.25,
                    "bookmaker_count": 1 if odds else 0,
                    "resultado_real": None if estado not in ESTADOS_FINALIZADOS else _resultado_desde_match({
                        "goals": {
                            "home": int(_texto_seguro(home.get("score") or 0, "0")),
                            "away": int(_texto_seguro(away.get("score") or 0, "0")),
                        }
                    }, estado),
                    "fuente_partidos": "espn_scoreboard",
                    "alineacion_confirmada": False,
                    "formacion_local": "",
                    "formacion_visitante": "",
                    "bajas_local": 0,
                    "bajas_visitante": 0,
                    "alerta_rotacion_local": False,
                    "alerta_rotacion_visitante": False,
                })

    if not partidos:
        return pd.DataFrame(), errors

    df = pd.DataFrame(partidos)
    df = df.sort_values(
        by=["prioridad_liga", "hora_partido", "liga", "local"],
        ascending=[False, True, True, True],
    ).drop_duplicates(subset=["fixture_id"], keep="first")

    return df.head(limit).reset_index(drop=True), errors

def _perfil_base_rapido():
    ataque = round(random.uniform(0.8, 2.2), 2)
    defensa = round(random.uniform(0.8, 2.0), 2)
    forma = random.randint(3, 10)
    return {
        "ataque": ataque,
        "defensa": defensa,
        "forma": forma,
        "racha": "",
        "ppm": round(max(forma / 5, 0.8), 2),
        "ppm_casa": 1.3,
        "ppm_fuera": 1.1,
    }


@lru_cache(maxsize=512)
def get_team_profile(team_id):

    url = f"https://v3.football.api-sports.io/fixtures?team={team_id}&last=10"

    try:
        res = requests.get(url, headers=headers, timeout=2.5).json()
        matches = res.get("response", [])

        if not matches:
            return _perfil_base_rapido()

        goles_favor = 0
        goles_contra = 0
        puntos = 0
        racha = []
        puntos_casa = partidos_casa = 0
        puntos_fuera = partidos_fuera = 0

        for m in matches:
            es_local = m["teams"]["home"]["id"] == team_id
            gf = (m["goals"]["home"] if es_local else m["goals"]["away"]) or 0
            ga = (m["goals"]["away"] if es_local else m["goals"]["home"]) or 0

            goles_favor += gf
            goles_contra += ga

            if gf > ga:
                puntos_partido = 3
                racha.append("W")
            elif gf == ga:
                puntos_partido = 1
                racha.append("D")
            else:
                puntos_partido = 0
                racha.append("L")

            puntos += puntos_partido
            if es_local:
                puntos_casa += puntos_partido
                partidos_casa += 1
            else:
                puntos_fuera += puntos_partido
                partidos_fuera += 1

        total_partidos = max(len(matches), 1)
        avg_favor = goles_favor / total_partidos
        avg_contra = goles_contra / total_partidos
        ppm = puntos / total_partidos
        ppm_casa = (puntos_casa / partidos_casa) if partidos_casa else ppm
        ppm_fuera = (puntos_fuera / partidos_fuera) if partidos_fuera else ppm

        return {
            "ataque": round(avg_favor, 2),
            "defensa": round(avg_contra, 2),
            "forma": puntos,
            "racha": "".join(racha[:5]),
            "ppm": round(ppm, 2),
            "ppm_casa": round(ppm_casa, 2),
            "ppm_fuera": round(ppm_fuera, 2),
        }

    except Exception:
        return _perfil_base_rapido()


@lru_cache(maxsize=512)
def get_team_stats(team_id):
    perfil = get_team_profile(team_id)
    return perfil["ataque"], perfil["defensa"], perfil["forma"]


def _parse_fecha_partido(fecha_raw):
    fecha_dt = pd.to_datetime(fecha_raw, utc=True, errors="coerce")
    if pd.isna(fecha_dt):
        return None
    return fecha_dt.tz_convert(TIMEZONE)


def _resultado_desde_match(match, estado):
    if estado not in ESTADOS_FINALIZADOS:
        return None

    gl = match["goals"]["home"]
    gv = match["goals"]["away"]

    if gl > gv:
        return "Gana local"
    if gv > gl:
        return "Gana visitante"
    return "Empate"


def _resumen_desde_match_api(match):
    fixture = match.get("fixture") or {}
    league = match.get("league") or {}
    teams = match.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    estado = ((fixture.get("status") or {}).get("short")) or ""
    fecha_match = _parse_fecha_partido(fixture.get("date"))
    return {
        "fixture_id": fixture.get("id"),
        "fecha_partido": fecha_match.strftime("%Y-%m-%d") if fecha_match else "",
        "hora_partido": fecha_match.strftime("%Y-%m-%d %H:%M") if fecha_match else "",
        "estado_partido": estado,
        "liga": league.get("name") or "",
        "partido": f"{home.get('name', '')} vs {away.get('name', '')}".strip(),
        "resultado_real": _resultado_desde_match(match, estado),
    }


def _buscar_resultado_api_por_fixture(fixture_id):
    if not fixture_id:
        return None
    try:
        response = requests.get(
            f"https://v3.football.api-sports.io/fixtures?id={fixture_id}",
            headers=headers,
            timeout=3,
        )
        data = response.json()
        matches = data.get("response") or []
        if not matches:
            return None
        return _resumen_desde_match_api(matches[0])
    except Exception:
        return None


def _buscar_resultado_api_por_fecha_partido(fecha, partido):
    if not fecha or not partido:
        return None
    try:
        response = requests.get(
            f"https://v3.football.api-sports.io/fixtures?date={fecha}",
            headers=headers,
            timeout=3,
        )
        data = response.json()
        matches = data.get("response") or []
    except Exception:
        return None

    mejor = None
    mejor_puntaje = 0.0
    partido = str(partido or "")
    try:
        local_objetivo, visitante_objetivo = [p.strip() for p in partido.split(" vs ", 1)]
    except ValueError:
        local_objetivo, visitante_objetivo = partido, ""

    for match in matches:
        home = ((match.get("teams") or {}).get("home") or {}).get("name") or ""
        away = ((match.get("teams") or {}).get("away") or {}).get("name") or ""
        puntaje = _puntaje_match_equipo(local_objetivo, home) + _puntaje_match_equipo(visitante_objetivo, away)
        if puntaje > mejor_puntaje:
            mejor_puntaje = puntaje
            mejor = match

    if mejor is None or mejor_puntaje < 1.2:
        return None

    return _resumen_desde_match_api(mejor)


def _buscar_resultado_espn_por_fecha_partido(fecha, partido):
    if not fecha or not partido:
        return None
    try:
        df_espn, _ = _fetch_matches_espn(fecha=fecha, limit=300)
    except Exception:
        return None

    if df_espn.empty:
        return None

    mejor = None
    mejor_puntaje = 0.0
    try:
        local_objetivo, visitante_objetivo = [p.strip() for p in str(partido).split(" vs ", 1)]
    except ValueError:
        local_objetivo, visitante_objetivo = str(partido), ""

    for _, row in df_espn.iterrows():
        puntaje = _puntaje_match_equipo(local_objetivo, row.get("local")) + _puntaje_match_equipo(visitante_objetivo, row.get("visitante"))
        if puntaje > mejor_puntaje:
            mejor_puntaje = puntaje
            mejor = row

    if mejor is None or mejor_puntaje < 1.2:
        return None

    return {
        "fixture_id": mejor.get("fixture_id"),
        "fecha_partido": mejor.get("fecha_partido") or "",
        "hora_partido": mejor.get("hora_partido") or "",
        "estado_partido": mejor.get("estado_partido") or "",
        "liga": mejor.get("liga") or "",
        "partido": f"{mejor.get('local', '')} vs {mejor.get('visitante', '')}".strip(),
        "resultado_real": mejor.get("resultado_real"),
    }


def buscar_resultado_partido(fecha=None, fixture_id=None, partido=None):
    if fixture_id:
        match = _buscar_resultado_api_por_fixture(fixture_id)
        if match:
            return match

    if fecha and partido:
        match = _buscar_resultado_api_por_fecha_partido(fecha, partido)
        if match:
            return match

        match = _buscar_resultado_espn_por_fecha_partido(fecha, partido)
        if match:
            return match

    df_cache = _cargar_cache_partidos()
    if df_cache.empty or not partido:
        return None

    cache_match = df_cache[(df_cache["local"] + " vs " + df_cache["visitante"]) == str(partido)]
    if cache_match.empty and fixture_id:
        cache_match = df_cache[df_cache["fixture_id"].astype(str) == str(fixture_id)]
    if cache_match.empty:
        return None

    row = cache_match.iloc[0]
    return {
        "fixture_id": row.get("fixture_id"),
        "fecha_partido": row.get("fecha_partido") or "",
        "hora_partido": row.get("hora_partido") or "",
        "estado_partido": row.get("estado_partido") or "",
        "liga": row.get("liga") or "",
        "partido": f"{row.get('local', '')} vs {row.get('visitante', '')}".strip(),
        "resultado_real": row.get("resultado_real"),
    }


def _cargar_stats_equipos(team_ids):
    stats = {}

    with ThreadPoolExecutor(max_workers=12) as executor:
        futuros = {
            executor.submit(get_team_profile, int(team_id)): int(team_id)
            for team_id in team_ids
        }

        for futuro in as_completed(futuros):
            team_id = futuros[futuro]
            try:
                stats[team_id] = futuro.result()
            except Exception:
                stats[team_id] = _perfil_base_rapido()

    return stats


def _stats_base_rapidos():
    return _perfil_base_rapido()


def _texto_seguro(valor, fallback=""):
    if valor is None:
        return fallback
    if isinstance(valor, float) and pd.isna(valor):
        return fallback
    return str(valor)


def _clasificar_liga(liga, pais=""):
    liga_txt = _texto_seguro(liga).lower()
    pais_txt = _texto_seguro(pais).lower()
    combinado = f"{liga_txt} {pais_txt}".strip()

    if any(x in combinado for x in [
        "u17", "u18", "u19", "u20", "u21", "u23",
        "reserves", "reserve", "amateur", "youth",
        "women", "femen", "femin",
        "division one", "division two", "2 division"
    ]):
        return False, 0, "Excluir"

    for liga_info in LIGAS_PRIORITARIAS:
        match_keyword = any(keyword in liga_txt for keyword in liga_info["keywords"])
        if not match_keyword:
            continue

        countries = liga_info.get("countries", [])
        if countries and pais_txt not in countries:
            continue

        return True, liga_info["priority"], liga_info["label"]

    if any(keyword in combinado for keyword in LIGAS_SECUNDARIAS):
        nombre_liga = _texto_seguro(liga, "Liga secundaria")
        return True, 55, nombre_liga

    return False, 0, "Excluir"


def _debe_incluir_partido(match_dt, estado, now_local, incluir_finalizados):
    if match_dt is None:
        return False

    if incluir_finalizados:
        return True

    if estado in ESTADOS_FINALIZADOS:
        return False

    limite_superior = now_local + pd.Timedelta(hours=48)

    if estado in ESTADOS_EN_CURSO:
        return match_dt.date() == now_local.date()

    return now_local <= match_dt <= limite_superior


def _debe_consultar_contexto_detallado(row_dict):
    if row_dict.get("fuente_partidos") != "api_football":
        return False
    prioridad = float(row_dict.get("prioridad_liga", 0) or 0)
    if prioridad < 86:
        return False

    estado = row_dict.get("estado_partido")
    if estado in ESTADOS_EN_CURSO:
        return True

    fecha_match = pd.to_datetime(row_dict.get("hora_partido"), errors="coerce")
    if pd.isna(fecha_match):
        return False

    ahora = datetime.now(TIMEZONE).replace(tzinfo=None)
    horas_faltantes = (fecha_match - ahora).total_seconds() / 3600
    return 0 <= horas_faltantes <= 18


def _debe_consultar_odds_api_football(row_dict):
    fixture_id = row_dict.get("fixture_id")
    if not fixture_id:
        return False
    prioridad = float(row_dict.get("prioridad_liga", 0) or 0)
    estado = str(row_dict.get("estado_partido") or "").upper()
    if estado in ESTADOS_EN_CURSO:
        return True
    return prioridad >= 86


@lru_cache(maxsize=256)
def _fetch_fixture_context_api(fixture_id, local=None, visitante=None):
    contexto = {
        "alineacion_confirmada": False,
        "once_confirmado_local": False,
        "once_confirmado_visitante": False,
        "formacion_local": "",
        "formacion_visitante": "",
        "suplentes_local": 0,
        "suplentes_visitante": 0,
        "bajas_local": 0,
        "bajas_visitante": 0,
        "bajas_suspension_local": 0,
        "bajas_suspension_visitante": 0,
        "bajas_lesion_local": 0,
        "bajas_lesion_visitante": 0,
        "banco_corto_local": False,
        "banco_corto_visitante": False,
        "alerta_rotacion_local": False,
        "alerta_rotacion_visitante": False,
    }
    if not fixture_id:
        return contexto

    local_norm = _normalizar_equipo(local)
    visitante_norm = _normalizar_equipo(visitante)

    try:
        lineups_data = requests.get(
            f"https://v3.football.api-sports.io/fixtures/lineups?fixture={fixture_id}",
            headers=headers,
            timeout=2.5,
        ).json()
        lineups = lineups_data.get("response") or []
    except Exception:
        lineups = []

    for lineup in lineups:
        team = lineup.get("team") or {}
        team_name = team.get("name") or ""
        team_norm = _normalizar_equipo(team_name)
        side = None
        if team_norm and team_norm == local_norm:
            side = "local"
        elif team_norm and team_norm == visitante_norm:
            side = "visitante"

        if not side:
            continue

        contexto["alineacion_confirmada"] = True
        contexto[f"formacion_{side}"] = lineup.get("formation") or ""
        titulares = lineup.get("startXI") or []
        suplentes = lineup.get("substitutes") or []
        contexto[f"once_confirmado_{side}"] = len(titulares) >= 11
        contexto[f"suplentes_{side}"] = len(suplentes)
        contexto[f"banco_corto_{side}"] = len(suplentes) < 8
        if len(suplentes) < 7:
            contexto[f"alerta_rotacion_{side}"] = True

    try:
        injuries_data = requests.get(
            f"https://v3.football.api-sports.io/injuries?fixture={fixture_id}",
            headers=headers,
            timeout=2.5,
        ).json()
        injuries = injuries_data.get("response") or []
    except Exception:
        injuries = []

    bajas_local = 0
    bajas_visitante = 0
    for item in injuries:
        team = item.get("team") or {}
        team_name = team.get("name") or ""
        team_norm = _normalizar_equipo(team_name)
        player = item.get("player") or {}
        reason = _texto_seguro(player.get("reason")).lower()
        if team_norm == local_norm:
            bajas_local += 1
            if "susp" in reason:
                contexto["bajas_suspension_local"] += 1
            else:
                contexto["bajas_lesion_local"] += 1
        elif team_norm == visitante_norm:
            bajas_visitante += 1
            if "susp" in reason:
                contexto["bajas_suspension_visitante"] += 1
            else:
                contexto["bajas_lesion_visitante"] += 1

    contexto["bajas_local"] = bajas_local
    contexto["bajas_visitante"] = bajas_visitante
    if bajas_local >= 4 or contexto["bajas_suspension_local"] >= 2:
        contexto["alerta_rotacion_local"] = True
    if bajas_visitante >= 4 or contexto["bajas_suspension_visitante"] >= 2:
        contexto["alerta_rotacion_visitante"] = True
    return contexto


@lru_cache(maxsize=512)
def _fetch_odds_fixture_api(fixture_id, live=False):
    if not fixture_id:
        return None
    endpoint = "odds/live" if live else "odds"
    try:
        data = requests.get(
            f"https://v3.football.api-sports.io/{endpoint}?fixture={fixture_id}",
            headers=headers,
            timeout=2.5,
        ).json()
        response = data.get("response") or []
        if not response:
            return None
        return _extraer_cuotas_api_football_item(response[0], live=live)
    except Exception:
        return None


def _cargar_contexto_fixtures_api(rows):
    contextos = {}
    candidatos = [
        row for row in rows
        if _debe_consultar_contexto_detallado(row)
    ][:MAX_FIXTURES_CONTEXTO]

    if not candidatos:
        return contextos

    with ThreadPoolExecutor(max_workers=4) as executor:
        futuros = {
            executor.submit(
                _fetch_fixture_context_api,
                row.get("fixture_id"),
                row.get("local"),
                row.get("visitante"),
            ): row.get("fixture_id")
            for row in candidatos
        }
        for futuro in as_completed(futuros):
            fixture_id = futuros[futuro]
            try:
                contextos[fixture_id] = futuro.result()
            except Exception:
                contextos[fixture_id] = {
                    "alineacion_confirmada": False,
                    "formacion_local": "",
                    "formacion_visitante": "",
                    "bajas_local": 0,
                    "bajas_visitante": 0,
                    "alerta_rotacion_local": False,
                    "alerta_rotacion_visitante": False,
                }
    return contextos


def _cargar_odds_fixtures_api(rows):
    odds_por_fixture = {}
    candidatos = [
        row for row in rows
        if _debe_consultar_odds_api_football(row)
    ][:MAX_FIXTURES_ODDS_API]

    if not candidatos:
        return odds_por_fixture

    with ThreadPoolExecutor(max_workers=4) as executor:
        futuros = {
            executor.submit(
                _fetch_odds_fixture_api,
                row.get("fixture_id"),
                str(row.get("estado_partido") or "").upper() in ESTADOS_EN_CURSO,
            ): row.get("fixture_id")
            for row in candidatos
        }
        for futuro in as_completed(futuros):
            fixture_id = futuros[futuro]
            try:
                odds_por_fixture[fixture_id] = futuro.result()
            except Exception:
                odds_por_fixture[fixture_id] = None
    return odds_por_fixture


def get_matches_api(fecha=None, limit=20, incluir_finalizados=None):
    limit = min(limit, FAST_LIMIT) if fecha is None else limit

    consulta_por_fecha = fecha is not None
    incluir_finalizados = (
        consulta_por_fecha if incluir_finalizados is None else incluir_finalizados
    )
    _set_last_api_status(True, "")

    if consulta_por_fecha:
        fechas_consulta = [fecha]
    else:
        now_local = datetime.now(TIMEZONE)
        fechas_consulta = [
            now_local.strftime("%Y-%m-%d"),
            (now_local + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            (now_local + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
        ]

    partidos_filtrados = []
    now_local = datetime.now(TIMEZONE)
    api_errors = []

    try:
        for fecha_consulta in fechas_consulta:
            url = f"https://v3.football.api-sports.io/fixtures?date={fecha_consulta}"
            response = requests.get(url, headers=headers, timeout=3)
            data = response.json()

            if data.get("errors"):
                error_text = json.dumps(data.get("errors"), ensure_ascii=False)
                api_errors.append(f"{fecha_consulta}: {error_text}")

            if "response" not in data or len(data["response"]) == 0:
                continue

            for match in data["response"]:
                estado = match["fixture"]["status"]["short"]
                fecha_match = _parse_fecha_partido(match["fixture"]["date"])

                if not _debe_incluir_partido(
                    fecha_match,
                    estado,
                    now_local,
                    incluir_finalizados
                ):
                    continue

                liga = match["league"]["name"]
                pais_liga = match["league"].get("country", "")
                incluir_liga, prioridad_liga, grupo_liga = _clasificar_liga(liga, pais_liga)
                if not incluir_liga:
                    continue

                local = match["teams"]["home"]["name"]
                visitante = match["teams"]["away"]["name"]

                local_id = match["teams"]["home"]["id"]
                visitante_id = match["teams"]["away"]["id"]

                partidos_filtrados.append({
                    "fixture_id": match["fixture"]["id"],
                    "fecha_partido": fecha_match.strftime("%Y-%m-%d") if fecha_match else "",
                    "hora_partido": fecha_match.strftime("%Y-%m-%d %H:%M") if fecha_match else "",
                    "estado_partido": estado,
                    "liga": liga,
                    "pais_liga": pais_liga,
                    "grupo_liga": grupo_liga,
                    "prioridad_liga": prioridad_liga,
                    "local": local,
                    "visitante": visitante,
                    "logo_local": match["teams"]["home"].get("logo"),
                    "logo_visitante": match["teams"]["away"].get("logo"),
                    "marcador_local": int(match["goals"]["home"] or 0),
                    "marcador_visitante": int(match["goals"]["away"] or 0),
                    "minuto_partido": _minuto_api_football(match),
                    "local_id": local_id,
                    "visitante_id": visitante_id,
                    "localia": 1,
                    "resultado_real": _resultado_desde_match(match, estado),
                    "record_local": "",
                    "record_visitante": "",
                    "goleador_local": "",
                    "goles_goleador_local": 0,
                    "goleador_visitante": "",
                    "goles_goleador_visitante": 0,
                })

        partidos = []
        if partidos_filtrados:
            df = pd.DataFrame(partidos_filtrados)
            for col in ["hora_partido", "liga", "local", "grupo_liga", "pais_liga"]:
                if col in df.columns:
                    df[col] = df[col].apply(_texto_seguro)
            if "prioridad_liga" not in df.columns:
                df["prioridad_liga"] = 0
            df = df.sort_values(
                by=["prioridad_liga", "hora_partido", "liga", "local"],
                ascending=[False, True, True, True]
            ).drop_duplicates(
                subset=["fixture_id"],
                keep="last"
            )
            df = df.head(limit).reset_index(drop=True)
            contextos_fixture = _cargar_contexto_fixtures_api(df.to_dict(orient="records")) if fecha is None else {}
            odds_fixture_api = _cargar_odds_fixtures_api(df.to_dict(orient="records")) if fecha is None else {}

            if fecha is None:
                team_ids = set(df["local_id"].astype(int)).union(set(df["visitante_id"].astype(int)))
                stats_equipos = _cargar_stats_equipos(team_ids)
            else:
                stats_equipos = {}

            for _, row in df.iterrows():
                if fecha is None:
                    perfil_local = stats_equipos.get(int(row["local_id"]), _stats_base_rapidos())
                    perfil_vis = stats_equipos.get(int(row["visitante_id"]), _stats_base_rapidos())
                else:
                    perfil_local = _stats_base_rapidos()
                    perfil_vis = _stats_base_rapidos()

                atk_local = perfil_local["ataque"]
                def_local = perfil_local["defensa"]
                forma_local = perfil_local["forma"]
                atk_vis = perfil_vis["ataque"]
                def_vis = perfil_vis["defensa"]
                forma_vis = perfil_vis["forma"]

                goles_local = max((atk_local * 0.9) - (def_vis * 0.6), 0.3)
                goles_visitante = max((atk_vis * 0.9) - (def_local * 0.6), 0.3)

                goles_local *= 1.10
                goles_local *= (1 + forma_local / 30)
                goles_visitante *= (1 + forma_vis / 30)

                diferencia = goles_local - goles_visitante
                goles_local += diferencia * 0.3
                goles_visitante -= diferencia * 0.3

                odds = get_odds(row["local"], row["visitante"]) if fecha is None else None
                odds_api_football = odds_fixture_api.get(row["fixture_id"]) if fecha is None else None
                if odds:
                    cuota_local = odds["cuota_local"]
                    cuota_visitante = odds["cuota_visitante"]
                    cuota_empate = odds["cuota_empate"]
                    fuente_cuotas = "the_odds_api"
                    market_match_score = float(odds.get("market_match_score", 0.72) or 0.72)
                    bookmaker_count = int(odds.get("bookmaker_count", 1) or 1)
                elif odds_api_football:
                    cuota_local = odds_api_football["cuota_local"]
                    cuota_visitante = odds_api_football["cuota_visitante"]
                    cuota_empate = odds_api_football["cuota_empate"]
                    fuente_cuotas = "api_football_odds_live" if str(row.get("estado_partido") or "").upper() in ESTADOS_EN_CURSO else "api_football_odds"
                    market_match_score = 0.93
                    bookmaker_count = 1
                else:
                    cuota_local = 2.2
                    cuota_visitante = 2.8
                    cuota_empate = 3.2
                    fuente_cuotas = "fallback"
                    market_match_score = 0.25
                    bookmaker_count = 0
                contexto_fixture = contextos_fixture.get(row["fixture_id"], {})

                partidos.append({
                    "fixture_id": row["fixture_id"],
                    "fecha_partido": row["fecha_partido"],
                    "hora_partido": row["hora_partido"],
                    "estado_partido": row["estado_partido"],
                    "liga": row["liga"],
                    "pais_liga": row.get("pais_liga"),
                    "grupo_liga": row.get("grupo_liga"),
                    "prioridad_liga": row.get("prioridad_liga", 0),
                    "local": row["local"],
                    "visitante": row["visitante"],
                    "logo_local": row.get("logo_local"),
                    "logo_visitante": row.get("logo_visitante"),
                    "marcador_local": row.get("marcador_local", 0),
                    "marcador_visitante": row.get("marcador_visitante", 0),
                    "minuto_partido": row.get("minuto_partido", ""),
                    "goles_local": round(goles_local, 2),
                    "goles_visitante": round(goles_visitante, 2),
                    "forma_local": forma_local,
                    "forma_visitante": forma_vis,
                    "racha_local": perfil_local.get("racha", ""),
                    "racha_visitante": perfil_vis.get("racha", ""),
                    "ppm_local": perfil_local.get("ppm", 0.0),
                    "ppm_visitante": perfil_vis.get("ppm", 0.0),
                    "ppm_local_casa": perfil_local.get("ppm_casa", perfil_local.get("ppm", 0.0)),
                    "ppm_visitante_fuera": perfil_vis.get("ppm_fuera", perfil_vis.get("ppm", 0.0)),
                    "record_local": row.get("record_local", ""),
                    "record_visitante": row.get("record_visitante", ""),
                    "goleador_local": row.get("goleador_local", ""),
                    "goles_goleador_local": row.get("goles_goleador_local", 0),
                    "goleador_visitante": row.get("goleador_visitante", ""),
                    "goles_goleador_visitante": row.get("goles_goleador_visitante", 0),
                    "localia": row["localia"],
                    "cuota_local": cuota_local,
                    "cuota_empate": cuota_empate,
                    "cuota_visitante": cuota_visitante,
                    "fuente_cuotas": fuente_cuotas,
                    "market_match_score": market_match_score,
                    "bookmaker_count": bookmaker_count,
                    "resultado_real": row["resultado_real"],
                    "fuente_partidos": "api_football",
                    "alineacion_confirmada": contexto_fixture.get("alineacion_confirmada", False),
                    "once_confirmado_local": contexto_fixture.get("once_confirmado_local", False),
                    "once_confirmado_visitante": contexto_fixture.get("once_confirmado_visitante", False),
                    "formacion_local": contexto_fixture.get("formacion_local", ""),
                    "formacion_visitante": contexto_fixture.get("formacion_visitante", ""),
                    "suplentes_local": contexto_fixture.get("suplentes_local", 0),
                    "suplentes_visitante": contexto_fixture.get("suplentes_visitante", 0),
                    "bajas_local": contexto_fixture.get("bajas_local", 0),
                    "bajas_visitante": contexto_fixture.get("bajas_visitante", 0),
                    "bajas_suspension_local": contexto_fixture.get("bajas_suspension_local", 0),
                    "bajas_suspension_visitante": contexto_fixture.get("bajas_suspension_visitante", 0),
                    "bajas_lesion_local": contexto_fixture.get("bajas_lesion_local", 0),
                    "bajas_lesion_visitante": contexto_fixture.get("bajas_lesion_visitante", 0),
                    "banco_corto_local": contexto_fixture.get("banco_corto_local", False),
                    "banco_corto_visitante": contexto_fixture.get("banco_corto_visitante", False),
                    "alerta_rotacion_local": contexto_fixture.get("alerta_rotacion_local", False),
                    "alerta_rotacion_visitante": contexto_fixture.get("alerta_rotacion_visitante", False),
                })

        df_partidos = pd.DataFrame(partidos)
        if not df_partidos.empty:
            df_partidos = _enriquecer_odds_desde_espn(df_partidos, fecha=fecha)
            _guardar_cache_partidos(df_partidos)
            if api_errors:
                _set_last_api_status(
                    True,
                    "Datos cargados con advertencias de la API principal.",
                    " | ".join(api_errors),
                    used_cache=False,
                )
            else:
                _set_last_api_status(True, "Datos cargados desde API-Football.", "", used_cache=False)
            return df_partidos

        df_espn, espn_errors = _fetch_matches_espn(fecha=fecha, limit=limit)
        if not df_espn.empty:
            _guardar_cache_partidos(df_espn)
            _set_last_api_status(
                True,
                "Se cargaron partidos usando ESPN como respaldo.",
                " | ".join(api_errors + espn_errors) if (api_errors or espn_errors) else "",
                used_cache=False,
                source="espn_scoreboard",
            )
            return df_espn

        df_cache = _cargar_cache_partidos()
        if not df_cache.empty:
            _set_last_api_status(
                False,
                "La API principal no devolvió partidos. Se cargó el último cache local disponible.",
                " | ".join(api_errors + espn_errors) if (api_errors or espn_errors) else "Sin respuesta útil de los proveedores.",
                used_cache=True,
                source="cache_local",
            )
            return df_cache.head(limit).reset_index(drop=True)

        _set_last_api_status(
            False,
            "No fue posible cargar partidos desde los proveedores en vivo y no hay cache local disponible.",
            " | ".join(api_errors + espn_errors) if (api_errors or espn_errors) else "Sin respuesta útil de los proveedores.",
            used_cache=False,
            source="providers_unavailable",
        )
        return pd.DataFrame()

    except Exception as exc:
        df_espn, espn_errors = _fetch_matches_espn(fecha=fecha, limit=limit)
        if not df_espn.empty:
            _guardar_cache_partidos(df_espn)
            _set_last_api_status(
                True,
                "Fallo la API principal. Se cargaron partidos usando ESPN como respaldo.",
                " | ".join([str(exc)] + espn_errors),
                used_cache=False,
                source="espn_scoreboard",
            )
            return df_espn

        df_cache = _cargar_cache_partidos()
        if not df_cache.empty:
            _set_last_api_status(
                False,
                "Fallo la carga en vivo. Se usó cache local.",
                str(exc),
                used_cache=True,
                source="cache_local",
            )
            return df_cache.head(limit).reset_index(drop=True)
        _set_last_api_status(
            False,
            "Fallo la carga en vivo y no hay cache local.",
            str(exc),
            used_cache=False,
            source="providers_unavailable",
        )
        return pd.DataFrame()
