import numpy as np
import pandas as pd
from math import exp, factorial


def detectar_trampa(prob, cuota):

    prob_casa = 1 / cuota
    edge = prob - prob_casa

    if prob > 0.60 and cuota > 2.2:
        return True, "Favorito con cuota inflada"

    if edge > 0.15:
        return True, "Edge exagerado (posible error o trampa)"

    if 0.45 < prob < 0.55 and (cuota < 1.8 or cuota > 2.8):
        return True, "Cuotas desbalanceadas en partido parejo"

    return False, ""


def clasificar_partido(row):

    razones = []

    if row["confianza"] < 60:
        razones.append("Baja confianza")

    if abs(row["prob_local"] - row["prob_visitante"]) < 10:
        razones.append("Partido equilibrado")

    if (row["value_local"] != "🟢 VALUE" and 
        row["value_visitante"] != "🟢 VALUE"):
        razones.append("Sin value")

    if len(razones) >= 2:
        return "NO APOSTAR", razones
    else:
        return "APOSTABLE", razones


def poisson_pmf(k, lam):
    return (lam**k * exp(-lam)) / factorial(k)


def normalizar_forma(forma):
    try:
        forma = float(forma)
    except (TypeError, ValueError):
        return 0.0

    # Convierte puntos recientes a un ajuste suave entre -0.18 y +0.18
    forma_centrada = (forma - 15) / 15
    return max(min(forma_centrada * 0.18, 0.18), -0.18)


def estimar_lambdas(row):
    base_local = max(float(row["goles_local"]), 0.2)
    base_visitante = max(float(row["goles_visitante"]), 0.2)

    ajuste_forma_local = normalizar_forma(row.get("forma_local", 15))
    ajuste_forma_visitante = normalizar_forma(row.get("forma_visitante", 15))
    ventaja_localia = 0.18 if row.get("localia", 1) else 0.0

    lam_local = base_local * (1 + ajuste_forma_local) + ventaja_localia
    lam_visitante = base_visitante * (1 + ajuste_forma_visitante)

    return max(lam_local, 0.2), max(lam_visitante, 0.2)


def matriz_probabilidades(lam_local, lam_visitante, max_goles=None):
    if max_goles is None:
        max_lambda = max(lam_local, lam_visitante)
        max_goles = max(6, int(np.ceil(max_lambda + 5)))

    matriz = np.zeros((max_goles + 1, max_goles + 1))

    for i in range(max_goles + 1):
        for j in range(max_goles + 1):
            matriz[i, j] = poisson_pmf(i, lam_local) * poisson_pmf(j, lam_visitante)

    total = matriz.sum()
    if total > 0:
        matriz = matriz / total

    return matriz


def calcular_1x2(matriz):
    prob_local = np.tril(matriz, -1).sum()
    prob_empate = np.trace(matriz)
    prob_visitante = np.triu(matriz, 1).sum()
    return prob_local, prob_empate, prob_visitante


def calcular_over_under(matriz, linea=2.5):
    over = 0
    under = 0

    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if i + j > linea:
                over += matriz[i, j]
            else:
                under += matriz[i, j]

    return over, under


def calcular_value(prob, cuota):

    prob_casa = 1 / cuota
    edge = prob - prob_casa

    if edge > 0.10:
        return "🔥 VALUE ALTO"
    elif edge > 0.04:
        return "🟢 VALUE"
    elif edge > 0:
        return "🟡 VALUE BAJO"
    else:
        return "🔴 NO VALUE"


def calcular_confianza(prob):
    if prob > 0.7:
        return 85
    elif prob > 0.6:
        return 75
    elif prob > 0.55:
        return 65
    else:
        return 50


def calcular_confianza_ajustada(prob, edge, bonus=0):
    confianza_base = calcular_confianza(prob)

    if edge > 0.12:
        confianza_base += 8
    elif edge > 0.08:
        confianza_base += 5
    elif edge < 0.04:
        confianza_base -= 5

    confianza_base += bonus

    return max(40, min(confianza_base, 92))


def calcular_stake_kelly(prob, cuota, factor=1.0):

    b = cuota - 1
    p = prob
    q = 1 - p

    kelly = (b * p - q) / b
    kelly = kelly * 0.5 * factor

    if kelly <= 0:
        return "❌ NO APOSTAR", 0

    if kelly < 0.01:
        return "🟡 MUY BAJO (0.5-1%)", 1
    elif kelly < 0.03:
        return "🟢 BAJO (1-2%)", 2
    elif kelly < 0.06:
        return "🟢 MEDIO (2-4%)", 3
    else:
        return "🔥 ALTO (4-6%)", 5


def clasificar_banda_confianza(confianza):
    if confianza >= 80:
        return "elite"
    if confianza >= 70:
        return "alta"
    if confianza >= 60:
        return "media"
    return "baja"


def clasificar_banda_edge(edge_pct):
    if edge_pct >= 10:
        return "elite"
    if edge_pct >= 6:
        return "alta"
    if edge_pct >= 3:
        return "media"
    return "baja"


def _obtener_ajuste_segmento(perfiles, clave):
    if not perfiles or not clave:
        return {}
    return perfiles.get(clave, {}) if isinstance(perfiles.get(clave, {}), dict) else {}


def calcular_ajustes_contexto(row, calibracion):
    perfiles = calibracion.get("segmentos", {}) if isinstance(calibracion, dict) else {}

    liga_mercado = None
    grupo_liga_mercado = None
    if row.get("liga") or row.get("mercado_historial"):
        liga_mercado = f"{row.get('liga', 'Sin dato') if row.get('liga') else 'Sin dato'} | {row.get('mercado_historial', 'Sin dato') if row.get('mercado_historial') else 'Sin dato'}"
    if row.get("grupo_liga") or row.get("mercado_historial"):
        grupo_liga_mercado = f"{row.get('grupo_liga', 'Sin dato') if row.get('grupo_liga') else 'Sin dato'} | {row.get('mercado_historial', 'Sin dato') if row.get('mercado_historial') else 'Sin dato'}"

    ajustes = {
        "edge_delta": 0.0,
        "confianza_delta": 0,
        "stake_factor": 1.0,
        "aprendizaje_score": 0.0,
        "aprendizaje_favorable": False,
        "aprendizaje_desfavorable": False,
        "motivos": [],
    }

    for tipo, valor in [
        ("liga_mercado", liga_mercado),
        ("grupo_liga_mercado", grupo_liga_mercado),
        ("liga", row.get("liga")),
        ("grupo_liga", row.get("grupo_liga")),
        ("mercado", row.get("mercado_historial")),
        ("fuente_cuotas", row.get("fuente_cuotas")),
        ("contexto_partido", row.get("contexto_partido")),
        ("prediccion_simple", row.get("prediccion_simple")),
        ("decision_simple", row.get("decision_simple")),
        ("ai_decision", row.get("ai_decision")),
        ("banda_confianza", row.get("banda_confianza")),
        ("banda_edge", row.get("banda_edge")),
    ]:
        perfil = _obtener_ajuste_segmento(perfiles.get(tipo, {}), valor)
        if not perfil:
            continue

        ajustes["edge_delta"] += perfil.get("edge_delta", 0.0)
        ajustes["confianza_delta"] += perfil.get("confianza_delta", 0)
        ajustes["stake_factor"] *= perfil.get("stake_factor", 1.0)
        risk_level = perfil.get("risk_level", "neutro")
        sample_size = int(perfil.get("sample_size", 0) or 0)
        if sample_size >= 6:
            if risk_level == "favorable":
                ajustes["aprendizaje_score"] += 1.2
            elif risk_level == "medio":
                ajustes["aprendizaje_score"] -= 0.25
            elif risk_level == "alto":
                ajustes["aprendizaje_score"] -= 1.0
        elif sample_size >= 4:
            if risk_level == "favorable":
                ajustes["aprendizaje_score"] += 0.4
            elif risk_level == "alto":
                ajustes["aprendizaje_score"] -= 0.4
        ajustes["motivos"].append(
            f"{tipo}: {valor} ({risk_level})"
        )

    if row.get("fuente_cuotas") == "fallback":
        ajustes["edge_delta"] += 0.015
        ajustes["confianza_delta"] -= 5
        ajustes["stake_factor"] *= 0.7
        ajustes["aprendizaje_score"] -= 0.4
        ajustes["motivos"].append("cuotas fallback")

    prioridad = row.get("prioridad_liga", 0) or 0
    if prioridad < 85:
        ajustes["edge_delta"] += 0.005
        ajustes["confianza_delta"] -= 2
        ajustes["stake_factor"] *= 0.9
        ajustes["aprendizaje_score"] -= 0.15
        ajustes["motivos"].append("liga fuera del grupo premium")

    ajustes["aprendizaje_favorable"] = ajustes["aprendizaje_score"] >= 1.0
    ajustes["aprendizaje_desfavorable"] = ajustes["aprendizaje_score"] <= -1.6
    return ajustes


def _calidad_mercado(row_dict):
    fuente = row_dict.get("fuente_cuotas")
    try:
        match_score = float(row_dict.get("market_match_score", 0) or 0)
    except Exception:
        match_score = 0.0
    try:
        bookmakers = int(float(row_dict.get("bookmaker_count", 0) or 0))
    except Exception:
        bookmakers = 0

    if fuente in {"api_football_odds", "api_football_odds_live"}:
        return "fuerte", 2.0
    if fuente == "espn_odds":
        return "aceptable", 1.0
    if fuente == "the_odds_api":
        if match_score >= 0.92 and bookmakers >= 2:
            return "fuerte", 1.8
        if match_score >= 0.78:
            return "aceptable", 1.0
        return "debil", -1.0
    if fuente == "fallback":
        return "debil", -2.0
    return "debil", -1.0


def calcular_ai_score(row_dict):
    score = 50.0

    ganador = row_dict.get("ganador")
    confianza = float(row_dict.get("confianza", 0) or 0)
    mejor_edge = float(row_dict.get("mejor_edge", 0) or 0)
    prioridad_liga = float(row_dict.get("prioridad_liga", 0) or 0)
    fuente_cuotas = row_dict.get("fuente_cuotas")
    estado = row_dict.get("estado")
    trampa = bool(row_dict.get("trampa"))
    contexto = _contexto_competitivo(row_dict)
    consenso = _consenso_analitico(row_dict)
    calidad_mercado, bonus_mercado = _calidad_mercado(row_dict)
    aprendizaje_score = float(row_dict.get("aprendizaje_score", 0) or 0)

    score += (confianza - 50) * 0.8
    score += mejor_edge * 220
    score += min(prioridad_liga / 8, 12)
    score += contexto["score"] * 3.5
    score += consenso["score"] * 3.0
    score += bonus_mercado
    score += aprendizaje_score * 1.8

    if ganador == "No bet":
        score -= 35
    if estado == "NO APOSTAR":
        score -= 18
    if trampa:
        score -= 30
    if fuente_cuotas in {"espn_odds", "api_football_odds", "api_football_odds_live"}:
        score += 4
    elif fuente_cuotas == "fallback":
        score -= 12
    if calidad_mercado == "debil":
        score -= 4
    if consenso["nivel"] == "bajo":
        score -= 6
    elif consenso["nivel"] == "alto":
        score += 4

    if row_dict.get("value_local") == "🔥 VALUE ALTO" or row_dict.get("value_visitante") == "🔥 VALUE ALTO":
        score += 8
    elif row_dict.get("value_empate") == "🔥 VALUE ALTO":
        score += 4
    elif (
        row_dict.get("value_local") == "🟢 VALUE" or
        row_dict.get("value_visitante") == "🟢 VALUE" or
        row_dict.get("value_empate") == "🟢 VALUE"
    ):
        score += 4

    if ganador == "Empate":
        score -= 8

    if ganador == "No bet" or float(row_dict.get("stake_num", 0) or 0) <= 0:
        score = min(score, 45)

    return max(0, min(round(score, 1), 100))


def etiquetar_ai_decision(ai_score):
    if ai_score >= 82:
        return "IA ELITE"
    if ai_score >= 70:
        return "IA FUERTE"
    if ai_score >= 58:
        return "IA OBSERVAR"
    return "IA DESCARTAR"


def resumir_ai_decision(row_dict):
    motivos = []
    calidad_mercado, _ = _calidad_mercado(row_dict)
    consenso = _consenso_analitico(row_dict)
    if calidad_mercado == "fuerte":
        motivos.append("mercado fuerte")
    elif calidad_mercado == "aceptable":
        motivos.append("mercado utilizable")
    else:
        motivos.append("mercado débil")

    if float(row_dict.get("mejor_edge", 0) or 0) >= 0.08:
        motivos.append("edge sólido")
    elif float(row_dict.get("mejor_edge", 0) or 0) >= 0.04:
        motivos.append("edge aceptable")
    else:
        motivos.append("edge corto")

    if float(row_dict.get("prioridad_liga", 0) or 0) >= 89:
        motivos.append("liga premium")
    elif float(row_dict.get("prioridad_liga", 0) or 0) >= 80:
        motivos.append("liga buena")
    else:
        motivos.append("liga secundaria")

    if row_dict.get("trampa"):
        motivos.append("señal de trampa")
    if consenso["nivel"] == "alto":
        motivos.append("señales alineadas")
    elif consenso["nivel"] == "bajo":
        motivos.append("señales en contra")
    elif consenso["nivel"] == "mixto":
        motivos.append("señales mixtas")

    return ", ".join(motivos)


def _prediccion_base(row_dict):
    prob_local = float(row_dict.get("prob_local", 0) or 0)
    prob_empate = float(row_dict.get("prob_empate", 0) or 0)
    prob_visitante = float(row_dict.get("prob_visitante", 0) or 0)

    if prob_local >= prob_empate and prob_local >= prob_visitante:
        return "Gana el local", prob_local
    if prob_visitante >= prob_local and prob_visitante >= prob_empate:
        return "Gana el visitante", prob_visitante
    return "Empate", prob_empate


def _ventaja_prediccion(row_dict):
    probs = sorted([
        float(row_dict.get("prob_local", 0) or 0),
        float(row_dict.get("prob_empate", 0) or 0),
        float(row_dict.get("prob_visitante", 0) or 0),
    ], reverse=True)
    if len(probs) < 2:
        return 0.0
    return round(probs[0] - probs[1], 1)


def _resumir_racha(valor):
    texto = str(valor or "").upper()
    if not texto:
        return ""
    wins = texto.count("W")
    draws = texto.count("D")
    losses = texto.count("L")
    return f"{wins}G-{draws}E-{losses}P"


def _equipo_en_mejor_momento(row_dict):
    try:
        ppm_local = float(row_dict.get("ppm_local_casa", row_dict.get("ppm_local", 0)) or 0)
        ppm_visitante = float(row_dict.get("ppm_visitante_fuera", row_dict.get("ppm_visitante", 0)) or 0)
    except Exception:
        ppm_local = 0.0
        ppm_visitante = 0.0

    if ppm_local <= 0:
        try:
            ppm_local = float(row_dict.get("forma_local", 0) or 0) / 10
        except Exception:
            ppm_local = 0.0
    if ppm_visitante <= 0:
        try:
            ppm_visitante = float(row_dict.get("forma_visitante", 0) or 0) / 10
        except Exception:
            ppm_visitante = 0.0

    if ppm_local <= 0 and ppm_visitante <= 0:
        try:
            goles_local = float(row_dict.get("goles_local", 0) or 0)
            goles_visitante = float(row_dict.get("goles_visitante", 0) or 0)
            diff_goles = round(abs(goles_local - goles_visitante), 2)
            if goles_local - goles_visitante > 0.2:
                return "local", diff_goles
            if goles_visitante - goles_local > 0.2:
                return "visitante", diff_goles
        except Exception:
            pass
        return None, 0.0

    diff = round(abs(ppm_local - ppm_visitante), 2)
    if ppm_local - ppm_visitante > 0.2:
        return "local", diff
    if ppm_visitante - ppm_local > 0.2:
        return "visitante", diff
    return None, diff


def _lado_predicho(prediccion):
    if prediccion == "Gana el local":
        return "local"
    if prediccion == "Gana el visitante":
        return "visitante"
    return "empate"


def _favorito_mercado(row_dict):
    cuotas = {
        "Gana el local": float(row_dict.get("cuota_local", 99) or 99),
        "Empate": float(row_dict.get("cuota_empate", 99) or 99),
        "Gana el visitante": float(row_dict.get("cuota_visitante", 99) or 99),
    }
    return min(cuotas, key=cuotas.get)


def _apoyo_mercado(row_dict):
    if row_dict.get("fuente_cuotas") not in {
        "the_odds_api",
        "espn_odds",
        "api_football_odds",
        "api_football_odds_live",
    }:
        return "sin_confirmacion"

    calidad_mercado, _ = _calidad_mercado(row_dict)
    if calidad_mercado == "debil":
        return "sin_confirmacion"

    prediccion, _ = _prediccion_base(row_dict)
    favorito_mercado = _favorito_mercado(row_dict)
    if favorito_mercado == prediccion:
        return "a_favor"
    if favorito_mercado == "Empate" and prediccion != "Empate":
        return "dudoso"
    return "en_contra"


def _ventaja_goleadora(row_dict):
    try:
        goles_local = int(float(row_dict.get("goles_goleador_local", 0) or 0))
        goles_visitante = int(float(row_dict.get("goles_goleador_visitante", 0) or 0))
    except Exception:
        return None, 0

    diff = abs(goles_local - goles_visitante)
    if diff < 3:
        return None, diff
    if goles_local > goles_visitante:
        return "local", diff
    if goles_visitante > goles_local:
        return "visitante", diff
    return None, diff


def _es_partido_en_vivo(row_dict):
    return str(row_dict.get("estado_partido") or "").upper() in {"LIVE", "HT", "1H", "2H"}


def _marcador_actual(row_dict):
    try:
        gl = int(float(row_dict.get("marcador_local", 0) or 0))
        gv = int(float(row_dict.get("marcador_visitante", 0) or 0))
    except Exception:
        return 0, 0
    return gl, gv


def _lado_marcador(row_dict):
    gl, gv = _marcador_actual(row_dict)
    if gl > gv:
        return "local"
    if gv > gl:
        return "visitante"
    return "empate"


def _impacto_bajas(row_dict):
    try:
        bajas_local = int(float(row_dict.get("bajas_local", 0) or 0))
        bajas_visitante = int(float(row_dict.get("bajas_visitante", 0) or 0))
        susp_local = int(float(row_dict.get("bajas_suspension_local", 0) or 0))
        susp_visitante = int(float(row_dict.get("bajas_suspension_visitante", 0) or 0))
        banco_corto_local = bool(row_dict.get("banco_corto_local"))
        banco_corto_visitante = bool(row_dict.get("banco_corto_visitante"))
    except Exception:
        return None, 0

    severidad_local = bajas_local + (susp_local * 0.7) + (0.8 if banco_corto_local else 0)
    severidad_visitante = bajas_visitante + (susp_visitante * 0.7) + (0.8 if banco_corto_visitante else 0)
    diff = round(abs(severidad_local - severidad_visitante), 1)
    if diff < 1.8:
        return None, diff
    if severidad_local > severidad_visitante:
        return "local", diff
    if severidad_visitante > severidad_local:
        return "visitante", diff
    return None, diff


def _contexto_competitivo(row_dict):
    prediccion, prob_top = _prediccion_base(row_dict)
    lado_predicho = _lado_predicho(prediccion)
    lado_momento, diff_ppm = _equipo_en_mejor_momento(row_dict)
    apoyo_mercado = _apoyo_mercado(row_dict)
    lado_goleador, diff_goleador = _ventaja_goleadora(row_dict)
    lado_bajas, diff_bajas = _impacto_bajas(row_dict)
    prioridad = float(row_dict.get("prioridad_liga", 0) or 0)
    ventaja = _ventaja_prediccion(row_dict)
    alineacion_confirmada = bool(row_dict.get("alineacion_confirmada"))
    once_confirmado_local = bool(row_dict.get("once_confirmado_local"))
    once_confirmado_visitante = bool(row_dict.get("once_confirmado_visitante"))
    formacion_local = row_dict.get("formacion_local") or ""
    formacion_visitante = row_dict.get("formacion_visitante") or ""
    alerta_rotacion_local = bool(row_dict.get("alerta_rotacion_local"))
    alerta_rotacion_visitante = bool(row_dict.get("alerta_rotacion_visitante"))
    banco_corto_local = bool(row_dict.get("banco_corto_local"))
    banco_corto_visitante = bool(row_dict.get("banco_corto_visitante"))

    score = 0
    alertas = []

    if lado_predicho != "empate" and lado_momento == lado_predicho:
        score += 2
    elif lado_predicho != "empate" and lado_momento and lado_momento != lado_predicho:
        score -= 2
        alertas.append("momento_en_contra")

    if apoyo_mercado == "a_favor":
        score += 2
    elif apoyo_mercado == "en_contra":
        score -= 2
        alertas.append("mercado_en_contra")
    elif apoyo_mercado == "dudoso":
        score -= 1

    if lado_predicho != "empate" and lado_goleador == lado_predicho:
        score += 1
    elif lado_predicho != "empate" and lado_goleador and lado_goleador != lado_predicho:
        score -= 1

    if lado_predicho != "empate" and lado_bajas:
        if lado_bajas == lado_predicho:
            score -= 2
            alertas.append("bajas_en_contra")
        else:
            score += 1

    if alineacion_confirmada:
        score += 0.5
    if once_confirmado_local and once_confirmado_visitante:
        score += 0.5

    if lado_predicho == "local":
        if alerta_rotacion_local or banco_corto_local:
            score -= 1.5
            alertas.append("rotacion_en_contra")
        elif alerta_rotacion_visitante or banco_corto_visitante:
            score += 1
    elif lado_predicho == "visitante":
        if alerta_rotacion_visitante or banco_corto_visitante:
            score -= 1.5
            alertas.append("rotacion_en_contra")
        elif alerta_rotacion_local or banco_corto_local:
            score += 1

    if formacion_local and formacion_visitante and formacion_local == formacion_visitante and ventaja < 5:
        alertas.append("planteo_similar")

    if prioridad >= 89 and ventaja < 5:
        alertas.append("duelo_top_apretado")

    if prob_top < 52 and ventaja < 4:
        alertas.append("lectura_fragil")

    return {
        "lado_predicho": lado_predicho,
        "lado_momento": lado_momento,
        "diff_ppm": diff_ppm,
        "apoyo_mercado": apoyo_mercado,
        "lado_goleador": lado_goleador,
        "diff_goleador": diff_goleador,
        "lado_bajas": lado_bajas,
        "diff_bajas": diff_bajas,
        "alineacion_confirmada": alineacion_confirmada,
        "once_confirmado_local": once_confirmado_local,
        "once_confirmado_visitante": once_confirmado_visitante,
        "formacion_local": formacion_local,
        "formacion_visitante": formacion_visitante,
        "alerta_rotacion_local": alerta_rotacion_local,
        "alerta_rotacion_visitante": alerta_rotacion_visitante,
        "banco_corto_local": banco_corto_local,
        "banco_corto_visitante": banco_corto_visitante,
        "score": score,
        "alertas": alertas,
    }


def _consenso_analitico(row_dict):
    contexto = _contexto_competitivo(row_dict)
    calidad_mercado, _ = _calidad_mercado(row_dict)
    live = _es_partido_en_vivo(row_dict)
    lado_predicho = contexto["lado_predicho"]
    lado_marcador = _lado_marcador(row_dict)

    a_favor = 0
    en_contra = 0
    notas = []

    if contexto["lado_momento"] == lado_predicho and lado_predicho != "empate":
        a_favor += 1
        notas.append("momento")
    elif contexto["lado_momento"] and contexto["lado_momento"] != lado_predicho and lado_predicho != "empate":
        en_contra += 1

    if contexto["apoyo_mercado"] == "a_favor":
        a_favor += 1
        notas.append("mercado")
    elif contexto["apoyo_mercado"] == "en_contra":
        en_contra += 1

    if contexto["lado_goleador"] == lado_predicho and lado_predicho != "empate":
        a_favor += 1
        notas.append("peso ofensivo")
    elif contexto["lado_goleador"] and contexto["lado_goleador"] != lado_predicho and lado_predicho != "empate":
        en_contra += 1

    if contexto["lado_bajas"] and lado_predicho != "empate":
        if contexto["lado_bajas"] == lado_predicho:
            en_contra += 1
        else:
            a_favor += 1
            notas.append("bajas rivales")

    if contexto["alineacion_confirmada"] and contexto["once_confirmado_local"] and contexto["once_confirmado_visitante"]:
        a_favor += 0.5
        notas.append("onces confirmados")

    if live and lado_predicho != "empate":
        if lado_marcador == lado_predicho:
            a_favor += 1
            notas.append("marcador acompaña")
        elif lado_marcador != "empate":
            en_contra += 1

    if calidad_mercado == "fuerte":
        a_favor += 0.5
    elif calidad_mercado == "debil":
        en_contra += 0.5

    score = round(a_favor - en_contra, 2)
    if score >= 2:
        nivel = "alto"
    elif score >= 0.75:
        nivel = "medio"
    elif score <= -1.5:
        nivel = "bajo"
    else:
        nivel = "mixto"

    return {
        "score": score,
        "nivel": nivel,
        "a_favor": a_favor,
        "en_contra": en_contra,
        "notas": notas[:3],
    }


def decision_simple(row_dict):
    if row_dict.get("trampa"):
        return "Evitar"

    prediccion, prob_top = _prediccion_base(row_dict)
    ventaja = _ventaja_prediccion(row_dict)
    mejor_edge_pct = float(row_dict.get("mejor_edge_pct", 0) or 0)
    ai_score = float(row_dict.get("ai_score", 0) or 0)
    confianza = float(row_dict.get("confianza", 0) or 0)
    estado_modelo = row_dict.get("estado")
    ganador_modelo = row_dict.get("ganador")
    contexto = _contexto_competitivo(row_dict)
    diff_ppm = contexto["diff_ppm"]
    live = _es_partido_en_vivo(row_dict)
    lado_predicho = contexto["lado_predicho"]
    lado_marcador = _lado_marcador(row_dict)
    gl, gv = _marcador_actual(row_dict)
    apoyo_mercado = contexto["apoyo_mercado"]
    alertas = set(contexto["alertas"])
    calidad_mercado, mercado_bonus = _calidad_mercado(row_dict)
    consenso = _consenso_analitico(row_dict)
    prioridad_liga = float(row_dict.get("prioridad_liga", 0) or 0)
    aprendizaje_score = float(row_dict.get("aprendizaje_score", 0) or 0)
    aprendizaje_favorable = bool(row_dict.get("aprendizaje_favorable", False))
    aprendizaje_desfavorable = bool(row_dict.get("aprendizaje_desfavorable", False))
    lectura_limpia = (
        lado_predicho != "empate" and
        "momento_en_contra" not in alertas and
        "bajas_en_contra" not in alertas and
        "lectura_fragil" not in alertas and
        "rotacion_en_contra" not in alertas
    )
    mercado_no_contradice = apoyo_mercado in {"a_favor", "dudoso", "sin_confirmacion"}
    contexto_fuerte = contexto["score"] >= 2
    ventaja_fuerte = ventaja >= 5
    premium_contexto = prioridad_liga >= 85 and diff_ppm >= 0.24
    puede_apostar = ganador_modelo != "No bet" and estado_modelo != "NO APOSTAR"
    mercado_aceptable = calidad_mercado in {"fuerte", "aceptable"}
    oportunidad_razonable = (
        prediccion != "Empate" and
        prob_top >= 53 and
        ventaja >= 4 and
        confianza >= 58 and
        ai_score >= 56 and
        contexto["score"] >= 2 and
        lectura_limpia and
        mercado_no_contradice
    )
    oportunidad_premium = (
        prediccion != "Empate" and
        prob_top >= 54 and
        ventaja >= 4.5 and
        confianza >= 56 and
        contexto["score"] >= 2 and
        lectura_limpia and
        premium_contexto and
        mercado_no_contradice
    )

    if (aprendizaje_desfavorable or consenso["nivel"] == "bajo") and calidad_mercado == "debil" and not live:
        return "Evitar"

    if prediccion == "Empate" and prob_top < 36:
        return "Evitar"

    if live:
        if lado_predicho == "empate":
            if gl == gv and prob_top >= 34:
                return "Mirar"
            return "Evitar"

        if lado_marcador == lado_predicho:
            if puede_apostar and gl == gv == 0 and contexto["score"] >= 2 and prob_top >= 56 and lectura_limpia and calidad_mercado != "debil":
                return "Apostar"
            if puede_apostar and abs(gl - gv) >= 1 and contexto["score"] >= 2 and prob_top >= 54 and lectura_limpia and mercado_no_contradice and consenso["score"] >= 1:
                return "Apostar"
            return "Mirar"

        if lado_marcador != "empate" and lado_marcador != lado_predicho:
            if abs(gl - gv) >= 2:
                return "Evitar"
            if contexto["score"] >= 2 and prob_top >= 53 and calidad_mercado != "debil":
                return "Mirar"
            return "Evitar"

        if gl == gv and contexto["score"] >= 1 and prob_top >= 52 and calidad_mercado != "debil":
            return "Mirar"

    if puede_apostar and (
        row_dict.get("ganador") != "No bet" and
        ai_score >= 68 and
        confianza >= 64 and
        contexto["score"] >= 2 and
        lectura_limpia and
        apoyo_mercado == "a_favor" and
        consenso["score"] >= 1.5
    ):
        return "Apostar"

    if puede_apostar and (
        estado_modelo == "APOSTABLE" and
        prediccion != "Empate" and
        prob_top >= 53 and
        mejor_edge_pct >= 6 and
        mercado_aceptable and
        lectura_limpia and
        mercado_no_contradice and
        contexto["score"] >= 1 and
        consenso["score"] >= 0.75
    ):
        return "Apostar"

    if puede_apostar and (
        prediccion != "Empate" and
        prob_top >= 56 and
        ventaja >= 5.5 and
        diff_ppm >= 0.24 and
        contexto["score"] >= 2 and
        lectura_limpia and
        calidad_mercado != "debil"
    ):
        return "Apostar"

    if puede_apostar and (
        prediccion != "Empate" and
        prob_top >= 55 and
        ventaja >= 5 and
        apoyo_mercado in {"a_favor", "dudoso"} and
        ai_score >= 62 and
        lectura_limpia and
        calidad_mercado != "debil" and
        consenso["score"] >= 0.75
    ):
        return "Apostar"

    if puede_apostar and (
        prediccion != "Empate" and
        prob_top >= 58 and
        ventaja >= 6.5 and
        contexto["score"] >= 3 and
        lectura_limpia and
        premium_contexto and
        mercado_no_contradice
    ):
        return "Apostar"

    if puede_apostar and (
        prediccion != "Empate" and
        prob_top >= 57 and
        ventaja >= 6 and
        contexto_fuerte and
        lectura_limpia and
        calidad_mercado == "debil" and
        prioridad_liga >= 90 and
        diff_ppm >= 0.3
    ):
        return "Apostar"

    if (
        row_dict.get("ganador") != "No bet" and
        estado_modelo == "APOSTABLE" and
        prediccion != "Empate" and
        confianza >= 64 and
        lectura_limpia and
        prioridad_liga >= 90 and
        apoyo_mercado != "en_contra"
    ):
        return "Apostar"

    if puede_apostar and mercado_aceptable and oportunidad_razonable:
        return "Apostar"

    if puede_apostar and calidad_mercado == "debil" and oportunidad_premium:
        return "Apostar"

    if puede_apostar and aprendizaje_favorable and (
        prediccion != "Empate" and
        prob_top >= 52 and
        ventaja >= 4 and
        confianza >= 56 and
        contexto["score"] >= 2 and
        lectura_limpia and
        mercado_no_contradice and
        consenso["score"] >= 1
    ):
        return "Apostar"

    if prediccion != "Empate" and prob_top >= 52 and ventaja >= 4 and contexto["score"] + mercado_bonus >= 0 and consenso["score"] >= 0:
        return "Mirar"

    if prob_top >= 50 and ai_score >= 48 and (mercado_aceptable or aprendizaje_score >= 0.8):
        return "Mirar"

    if prediccion != "Empate" and prob_top >= 49 and (ventaja >= 5 or diff_ppm >= 0.3 or contexto["score"] >= 1):
        return "Mirar"

    if estado_modelo == "APOSTABLE" and prediccion != "Empate":
        return "Mirar"

    return "Evitar"


def prediccion_simple(ganador):
    mapping = {
        "Gana local": "Gana el local",
        "Gana visitante": "Gana el visitante",
        "Empate": "Empate",
    }
    return mapping.get(ganador, "Partido para mirar")


def contexto_simple(row_dict):
    frases = []
    contexto = _contexto_competitivo(row_dict)
    live = _es_partido_en_vivo(row_dict)
    gl, gv = _marcador_actual(row_dict)

    if live:
        frases.append(f"marcador actual {gl}-{gv}")

    if contexto["apoyo_mercado"] == "a_favor":
        frases.append("el mercado acompaña la lectura")
    elif contexto["apoyo_mercado"] == "en_contra":
        frases.append("el mercado va en contra de la lectura")
    else:
        frases.append("sin apoyo claro del mercado")

    racha_local = row_dict.get("racha_local")
    racha_visitante = row_dict.get("racha_visitante")
    if racha_local or racha_visitante:
        resumen_local = _resumir_racha(racha_local)
        resumen_visitante = _resumir_racha(racha_visitante)
        if resumen_local or resumen_visitante:
            frases.append(f"racha reciente {resumen_local or '-'} vs {resumen_visitante or '-'}")

    lado_momento, diff_ppm = contexto["lado_momento"], contexto["diff_ppm"]
    if lado_momento == "local":
        frases.append("el local llega en mejor momento")
    elif lado_momento == "visitante":
        frases.append("el visitante llega en mejor momento")
    elif diff_ppm <= 0.2:
        frases.append("momento muy parejo")

    if contexto["lado_goleador"] == "local" and contexto["diff_goleador"] >= 3:
        frases.append("el local tiene más peso ofensivo")
    elif contexto["lado_goleador"] == "visitante" and contexto["diff_goleador"] >= 3:
        frases.append("el visitante tiene más peso ofensivo")
    elif contexto["lado_bajas"] == "local" and contexto["diff_bajas"] >= 2:
        frases.append("el local llega con más bajas")
    elif contexto["lado_bajas"] == "visitante" and contexto["diff_bajas"] >= 2:
        frases.append("el visitante llega con más bajas")
    elif contexto["alerta_rotacion_local"] and contexto["lado_predicho"] == "local":
        frases.append("el local no llega con un once tan limpio")
    elif contexto["alerta_rotacion_visitante"] and contexto["lado_predicho"] == "visitante":
        frases.append("el visitante no llega con un once tan limpio")
    elif contexto["alineacion_confirmada"] and contexto["once_confirmado_local"] and contexto["once_confirmado_visitante"]:
        frases.append("los onces ya están confirmados")
    elif row_dict.get("goles") == "Under 2.5":
        frases.append("se espera un partido cerrado")
    elif row_dict.get("goles") == "Over 2.5":
        frases.append("se espera un partido abierto")

    return ", ".join(frases[:3])


def razon_simple(row_dict):
    prediccion, prob_top = _prediccion_base(row_dict)
    ventaja = _ventaja_prediccion(row_dict)
    contexto = _contexto_competitivo(row_dict)
    lado_momento, diff_ppm = contexto["lado_momento"], contexto["diff_ppm"]
    mercado_ok = contexto["apoyo_mercado"] == "a_favor"
    live = _es_partido_en_vivo(row_dict)
    gl, gv = _marcador_actual(row_dict)
    lado_marcador = _lado_marcador(row_dict)

    if live:
        if row_dict.get("decision_simple") == "Apostar":
            return f"El partido va {gl}-{gv} y la lectura sigue favoreciendo a {prediccion.lower()}."
        if row_dict.get("decision_simple") == "Mirar":
            if lado_marcador == contexto["lado_predicho"]:
                return f"El marcador va {gl}-{gv} y la lectura del juego sigue en esa dirección."
            if gl == gv:
                return f"Con el {gl}-{gv} actual, el partido sigue abierto y merece seguimiento."
            return f"El partido va {gl}-{gv}; hay señales para seguirlo, pero no para entrar fuerte."
        if lado_marcador != "empate" and lado_marcador != contexto["lado_predicho"]:
            return f"El marcador actual va {gl}-{gv} en contra de la lectura previa; mejor no forzar apuesta."
        return f"Con el {gl}-{gv} actual no hay una lectura lo bastante limpia para recomendar entrada."

    if row_dict.get("decision_simple") == "Apostar":
        if prediccion == "Gana el local":
            if contexto["lado_bajas"] == "visitante":
                return f"El local queda mejor parado y además el rival llega más limitado ({round(prob_top,1)}%)."
            if lado_momento == "local":
                return f"El local llega mejor y el partido se inclina hacia su lado ({round(prob_top,1)}%)."
            return f"Hay una ventaja clara para el local y la lectura general acompaña ({round(prob_top,1)}%)."
        if prediccion == "Gana el visitante":
            if contexto["lado_bajas"] == "local":
                return f"El visitante queda mejor parado y el local llega más condicionado ({round(prob_top,1)}%)."
            if lado_momento == "visitante":
                return f"El visitante llega más fuerte y hoy parece tener ventaja ({round(prob_top,1)}%)."
            return f"El visitante tiene una señal bastante limpia para este partido ({round(prob_top,1)}%)."
        return f"El empate toma fuerza en un partido muy cerrado ({round(prob_top,1)}%)."

    if row_dict.get("decision_simple") == "Mirar":
        if prediccion == "Empate":
            return "Se perfila un partido muy cerrado, útil para seguirlo pero no para entrar fuerte."
        if contexto["apoyo_mercado"] == "en_contra":
            return f"Hay una lectura a favor de {prediccion.lower()}, pero el mercado manda una señal contraria."
        if not mercado_ok:
            return f"Hay una inclinación hacia {prediccion.lower()}, pero el mercado todavía no lo respalda del todo."
        if diff_ppm <= 0.2 or ventaja < 5:
            return f"Hay una ligera ventaja para {prediccion.lower()}, pero el partido sigue siendo delicado."
        return f"La lectura favorece a {prediccion.lower()}, aunque todavía no es una señal fuerte."

    if ventaja < 4:
        return "Partido muy parejo; mejor no precipitarse."
    if contexto["apoyo_mercado"] == "en_contra":
        return "El mercado y la lectura del partido no van en la misma dirección; mejor esperar."
    if contexto["lado_bajas"] == contexto["lado_predicho"] and contexto["diff_bajas"] >= 2:
        return "El equipo que parece favorito llega con más ausencias; mejor no entrar."
    if "rotacion_en_contra" in contexto["alertas"]:
        return "El equipo que mejor se veía no llega con un once lo bastante limpio; mejor esperar."
    if contexto["lado_predicho"] != "empate" and contexto["lado_momento"] and contexto["lado_momento"] != contexto["lado_predicho"]:
        return "El equipo que parece favorito no llega en mejor momento; mejor no entrar."
    if not mercado_ok:
        return "La lectura deportiva existe, pero sin respaldo claro del mercado es mejor esperar."
    return "La lectura no es lo bastante limpia para recomendar apuesta."


def disciplina_simple(row_dict):
    decision = row_dict.get("decision_simple", "Mirar")
    if decision == "Evitar":
        return "No tocar"

    contexto = _contexto_competitivo(row_dict)
    alertas = set(contexto["alertas"])
    apoyo_mercado = contexto["apoyo_mercado"]
    confianza = float(row_dict.get("confianza", 0) or 0)
    ai_score = float(row_dict.get("ai_score", 0) or 0)
    stake_num = float(row_dict.get("stake_num", 0) or 0)
    prediccion, prob_top = _prediccion_base(row_dict)
    ventaja = _ventaja_prediccion(row_dict)
    live = _es_partido_en_vivo(row_dict)
    gl, gv = _marcador_actual(row_dict)
    lado_marcador = _lado_marcador(row_dict)
    lado_predicho = contexto["lado_predicho"]
    calidad_mercado, _ = _calidad_mercado(row_dict)
    consenso = _consenso_analitico(row_dict)
    prioridad_liga = float(row_dict.get("prioridad_liga", 0) or 0)
    aprendizaje_score = float(row_dict.get("aprendizaje_score", 0) or 0)
    aprendizaje_favorable = bool(row_dict.get("aprendizaje_favorable", False))
    aprendizaje_desfavorable = bool(row_dict.get("aprendizaje_desfavorable", False))
    premium_contexto = prioridad_liga >= 85 and contexto["diff_ppm"] >= 0.24

    lectura_fuerte = (
        prediccion != "Empate" and
        apoyo_mercado == "a_favor" and
        contexto["score"] >= 3 and
        confianza >= 68 and
        ai_score >= 70 and
        stake_num >= 1.5 and
        ventaja >= 5 and
        calidad_mercado in {"fuerte", "aceptable"} and
        "momento_en_contra" not in alertas and
        "bajas_en_contra" not in alertas and
        "rotacion_en_contra" not in alertas and
        "lectura_fragil" not in alertas
    )

    lectura_pequena = (
        prediccion != "Empate" and
        contexto["score"] >= 1 and
        confianza >= 52 and
        ai_score >= 40 and
        ventaja >= 3.5 and
        calidad_mercado != "debil" and
        "lectura_fragil" not in alertas
    )
    lectura_apostable = (
        prediccion != "Empate" and
        contexto["score"] >= 2 and
        confianza >= 58 and
        ai_score >= 54 and
        ventaja >= 4 and
        "momento_en_contra" not in alertas and
        "bajas_en_contra" not in alertas and
        "rotacion_en_contra" not in alertas
    )
    mercado_verificado = apoyo_mercado == "a_favor"
    mercado_neutro = apoyo_mercado in {"a_favor", "dudoso"}

    if (aprendizaje_desfavorable or consenso["nivel"] == "bajo") and calidad_mercado == "debil":
        return "No tocar"

    if row_dict.get("ganador") == "No bet":
        if decision == "Mirar":
            if (
                prediccion != "Empate" and
                mercado_verificado and
                contexto["score"] >= 2 and
                confianza >= 55 and
                "rotacion_en_contra" not in alertas and
                "bajas_en_contra" not in alertas
            ):
                return "Entrada pequeña"
            return "Solo seguimiento"
        return "No tocar"

    if decision == "Apostar":
        if lectura_fuerte and not live:
            return "Apuesta fuerte"
        if lectura_apostable and (mercado_verificado or premium_contexto or aprendizaje_favorable) and consenso["score"] >= 0.75:
            return "Apuesta pequeña"
        if calidad_mercado in {"fuerte", "aceptable"} and contexto["score"] >= 2 and confianza >= 56:
            return "Apuesta pequeña"
        return "Entrada pequeña"

    if decision == "Mirar":
        if live:
            if (
                lado_predicho != "empate" and
                lado_marcador == lado_predicho and
                abs(gl - gv) >= 1 and
                contexto["score"] >= 1 and
                mercado_neutro and
                "lectura_fragil" not in alertas and
                "rotacion_en_contra" not in alertas and
                "bajas_en_contra" not in alertas and
                consenso["score"] >= 0.5
            ):
                return "Entrada pequeña"
            if (
                gl == gv and
                lado_predicho != "empate" and
                lectura_pequena and
                contexto["score"] >= 2 and
                mercado_verificado
            ):
                return "Entrada pequeña"
        if (
            lectura_pequena and
            prob_top >= 52 and
            (mercado_neutro or aprendizaje_score >= 1.0) and
            "momento_en_contra" not in alertas and
            "rotacion_en_contra" not in alertas and
            "bajas_en_contra" not in alertas and
            consenso["score"] >= 0.5
        ):
            return "Entrada pequeña"
        if (
            lectura_pequena and
            prob_top >= 55 and
            mercado_neutro and
            contexto["score"] >= 2 and
            calidad_mercado == "fuerte"
        ):
            return "Entrada pequeña"
        return "Solo seguimiento"

    return "No tocar"


def riesgo_simple(row_dict):
    disciplina = row_dict.get("disciplina_simple") or disciplina_simple(row_dict)
    contexto = _contexto_competitivo(row_dict)
    alertas = set(contexto["alertas"])

    if disciplina == "No tocar":
        return "Riesgo alto"
    if disciplina == "Solo seguimiento":
        return "Riesgo medio"
    if "bajas_en_contra" in alertas or "rotacion_en_contra" in alertas or "mercado_en_contra" in alertas:
        return "Riesgo medio"
    if disciplina == "Apuesta fuerte":
        return "Riesgo controlado"
    return "Riesgo medio"


def _ganador_desde_prediccion_simple(prediccion):
    mapping = {
        "Gana el local": "Gana local",
        "Gana el visitante": "Gana visitante",
        "Empate": "Empate",
    }
    return mapping.get(str(prediccion or "").strip(), "No bet")


def _promover_mejores_oportunidades(df):
    if df.empty or "decision_simple" not in df.columns:
        return df

    if (df["decision_simple"] == "Apostar").any():
        return df

    elegibles = df[
        (df["decision_simple"] == "Mirar") &
        (df["prediccion_simple"].isin(["Gana el local", "Gana el visitante"])) &
        (~df["trampa"].fillna(False)) &
        (df["mejor_edge_pct"].fillna(0) >= 5.0) &
        (df["prioridad_liga"].fillna(0) >= 70)
    ].copy()

    if elegibles.empty:
        return df

    elegibles["fuente_rank"] = elegibles["fuente_cuotas"].map({
        "api_football_odds": 3,
        "api_football_odds_live": 3,
        "the_odds_api": 3,
        "espn_odds": 2,
        "fallback": 1,
    }).fillna(0)
    elegibles["oportunidad_rank"] = (
        elegibles["fuente_rank"] * 100
        + elegibles["prioridad_liga"].fillna(0) * 0.7
        + elegibles["mejor_edge_pct"].fillna(0) * 2.5
        + elegibles["confianza"].fillna(0) * 0.4
        + elegibles["contexto_score"].fillna(0) * 8
    )
    elegibles = elegibles.sort_values(
        by=["oportunidad_rank", "mejor_edge_pct", "prioridad_liga"],
        ascending=[False, False, False],
    )

    max_promociones = 2 if len(elegibles) >= 3 else 1
    promos = elegibles.head(max_promociones)
    df_result = df.copy()

    for idx in promos.index:
        pred = df_result.at[idx, "prediccion_simple"]
        ganador = df_result.at[idx, "ganador"]
        if ganador == "No bet":
            nuevo_ganador = _ganador_desde_prediccion_simple(pred)
            df_result.at[idx, "ganador"] = nuevo_ganador
            goles = df_result.at[idx, "goles"] if "goles" in df_result.columns else "Línea dudosa"
            df_result.at[idx, "mercado"] = generar_mercado(nuevo_ganador, goles)
            df_result.at[idx, "mercado_historial"] = generar_mercado(nuevo_ganador, goles)

        df_result.at[idx, "decision_simple"] = "Apostar"
        df_result.at[idx, "disciplina_simple"] = "Apuesta pequeña"
        df_result.at[idx, "riesgo_simple"] = "Riesgo medio"
        razon = str(df_result.at[idx, "razon_simple"] or "").strip()
        prefijo = "Es la mejor oportunidad disponible del día. "
        if not razon.startswith(prefijo):
            df_result.at[idx, "razon_simple"] = prefijo + razon
        if float(df_result.at[idx, "stake_num"] or 0) <= 0:
            df_result.at[idx, "stake_num"] = 1
            df_result.at[idx, "stake"] = "🟢 BAJO (1-2%)"

    return df_result


def ajustar_stake_con_cap(stake_num, stake_label, cap):
    try:
        stake_num = float(stake_num)
    except Exception:
        stake_num = 0

    if stake_num <= 0:
        return "❌ NO APOSTAR", 0

    stake_num = min(stake_num, cap)

    if stake_num <= 1:
        return "🟡 MUY BAJO (0.5-1%)", 1
    if stake_num <= 2:
        return "🟢 BAJO (1-2%)", 2
    if stake_num <= 3:
        return "🟢 MEDIO (2-4%)", 3
    return "🔥 ALTO (4-6%)", stake_num


def seleccionar_cartera_ia(df, max_picks=5):
    if df.empty:
        return df

    picks = df[
        (df["ganador"] != "No bet") &
        (df["ai_score"] >= 70) &
        (df["stake_num"] > 0) &
        (~df["trampa"]) &
        (df["fuente_cuotas"].isin(["the_odds_api", "espn_odds", "api_football_odds", "api_football_odds_live"]))
    ].copy()

    if picks.empty:
        return picks

    picks = picks.sort_values(
        by=["ai_score", "confianza", "mejor_edge_pct"],
        ascending=[False, False, False],
    )

    seleccionados = []
    conteo_liga = {}

    for _, row in picks.iterrows():
        liga = row.get("liga", "Sin liga")
        if conteo_liga.get(liga, 0) >= 2:
            continue
        seleccionados.append(row.to_dict())
        conteo_liga[liga] = conteo_liga.get(liga, 0) + 1
        if len(seleccionados) >= max_picks:
            break

    return pd.DataFrame(seleccionados)


def construir_plan_banca(df, bankroll, riesgo_diario_pct=3.0, max_riesgo_pick_pct=1.0, max_picks=5):
    if df.empty:
        return pd.DataFrame(), {
            "bankroll": float(bankroll),
            "riesgo_diario": 0.0,
            "riesgo_por_pick": 0.0,
            "cartera_recomendada": 0,
            "riesgo_asignado": 0.0,
            "riesgo_restante": 0.0,
        }

    cartera = seleccionar_cartera_ia(df, max_picks=max_picks).copy()
    bankroll = max(float(bankroll or 0), 0.0)
    riesgo_diario = bankroll * (float(riesgo_diario_pct or 0) / 100)
    riesgo_por_pick = bankroll * (float(max_riesgo_pick_pct or 0) / 100)

    if cartera.empty or bankroll <= 0 or riesgo_diario <= 0 or riesgo_por_pick <= 0:
        return cartera, {
            "bankroll": bankroll,
            "riesgo_diario": round(riesgo_diario, 2),
            "riesgo_por_pick": round(riesgo_por_pick, 2),
            "cartera_recomendada": 0,
            "riesgo_asignado": 0.0,
            "riesgo_restante": round(riesgo_diario, 2),
        }

    cartera["peso_bruto"] = (
        cartera["ai_score"].clip(lower=1) *
        cartera["confianza"].clip(lower=1) *
        cartera["ev_por_unidad"].clip(lower=0.01)
    )
    total_peso = float(cartera["peso_bruto"].sum()) or 1.0
    cartera["pct_bankroll_sugerido"] = (
        (cartera["peso_bruto"] / total_peso) * (float(riesgo_diario_pct or 0))
    ).clip(upper=float(max_riesgo_pick_pct or 0))
    cartera["monto_sugerido"] = (cartera["pct_bankroll_sugerido"] / 100) * bankroll

    riesgo_asignado = float(cartera["monto_sugerido"].sum())
    if riesgo_asignado > riesgo_diario and riesgo_asignado > 0:
        factor_ajuste = riesgo_diario / riesgo_asignado
        cartera["monto_sugerido"] = cartera["monto_sugerido"] * factor_ajuste
        cartera["pct_bankroll_sugerido"] = (cartera["monto_sugerido"] / bankroll) * 100
        riesgo_asignado = float(cartera["monto_sugerido"].sum())

    cartera["prioridad_cartera"] = cartera["ai_score"].rank(method="dense", ascending=False).astype(int)
    cartera["monto_sugerido"] = cartera["monto_sugerido"].round(2)
    cartera["pct_bankroll_sugerido"] = cartera["pct_bankroll_sugerido"].round(2)

    return cartera.sort_values(by=["prioridad_cartera", "ai_score"], ascending=[True, False]), {
        "bankroll": round(bankroll, 2),
        "riesgo_diario": round(riesgo_diario, 2),
        "riesgo_por_pick": round(riesgo_por_pick, 2),
        "cartera_recomendada": int(len(cartera)),
        "riesgo_asignado": round(riesgo_asignado, 2),
        "riesgo_restante": round(max(riesgo_diario - riesgo_asignado, 0), 2),
    }


def generar_mercado(ganador, goles):
    if "Gana" in ganador and "Over" in goles:
        return "Favorito + Over"
    elif "Gana" in ganador and "Under" in goles:
        return "Favorito + Under"
    elif ganador == "Empate":
        return "Empate o Under"
    else:
        return "Partido cerrado"


def analizar_partidos(df, calibracion=None):
    calibracion = calibracion or {}
    min_edge = calibracion.get("min_edge", 0.03)
    confianza_bonus = calibracion.get("confianza_bonus", 0)
    stake_factor = calibracion.get("stake_factor", 1.0)
    max_stake_cap = calibracion.get("max_stake_cap", 5)

    resultados = []

    for _, row in df.iterrows():

        lam_local, lam_visitante = estimar_lambdas(row)

        matriz = matriz_probabilidades(lam_local, lam_visitante)

        prob_local, prob_empate, prob_visitante = calcular_1x2(matriz)
        over, under = calcular_over_under(matriz)

        edge_local = prob_local - (1 / row["cuota_local"])
        edge_visitante = prob_visitante - (1 / row["cuota_visitante"])
        edge_empate = prob_empate - (1 / row["cuota_empate"])

        edges = {
            "Local": edge_local,
            "Visitante": edge_visitante,
            "Empate": edge_empate
        }

        mejor = max(edges, key=edges.get)
        mejor_edge = edges[mejor]

        if mejor == "Local":
            prob = prob_local
        elif mejor == "Visitante":
            prob = prob_visitante
        else:
            prob = prob_empate

        mejor_edge_pct = round(mejor_edge * 100, 2)
        row["mercado_historial"] = generar_mercado(
            "Gana local" if mejor == "Local" else "Gana visitante" if mejor == "Visitante" else "Empate",
            "Over 2.5" if over > 0.55 else "Under 2.5" if under > 0.55 else "Línea dudosa",
        )
        row["contexto_partido"] = (
            "En vivo" if str(row.get("estado_partido", "")).upper() in {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}
            else "Prepartido" if str(row.get("estado_partido", "")).upper() in {"NS", "TBD", "PST"}
            else "Finalizado" if str(row.get("estado_partido", "")).upper() in {"FT", "AET", "PEN", "CANC", "ABD", "AWD", "WO"}
            else "Sin dato"
        )
        row["banda_edge"] = clasificar_banda_edge(mejor_edge_pct)
        row["banda_confianza"] = clasificar_banda_confianza(calcular_confianza(prob))
        row["prediccion_simple"] = (
            "Gana el local" if mejor == "Local" else
            "Gana el visitante" if mejor == "Visitante" else
            "Empate"
        )
        row["decision_simple"] = "Apostar" if mejor_edge > min_edge and prob > 0.50 else "Mirar"
        row["ai_decision"] = "IA FUERTE" if prob > 0.58 and mejor_edge > 0.04 else "IA OBSERVAR"

        ajustes_contexto = calcular_ajustes_contexto(row, calibracion)
        min_edge_ajustado = max(min_edge + ajustes_contexto["edge_delta"], 0.02)
        confianza_bonus_total = confianza_bonus + ajustes_contexto["confianza_delta"]
        stake_factor_total = stake_factor * ajustes_contexto["stake_factor"]

        if mejor == "Empate":
            min_edge_ajustado += 0.015

        if mejor_edge > min_edge_ajustado and ((mejor == "Empate" and prob > 0.28) or (mejor != "Empate" and prob > 0.50)):

            if mejor == "Local":
                ganador = "Gana local"
            elif mejor == "Visitante":
                ganador = "Gana visitante"
            else:
                ganador = "Empate"

        else:
            ganador = "No bet"

        trampa = False
        razon_trampa = ""

        if ganador == "Gana local":
            trampa, razon_trampa = detectar_trampa(prob_local, row["cuota_local"])

        elif ganador == "Gana visitante":
            trampa, razon_trampa = detectar_trampa(prob_visitante, row["cuota_visitante"])

        elif ganador == "Empate":
            trampa, razon_trampa = detectar_trampa(prob_empate, row["cuota_empate"])

        if over > 0.55:
            goles = "Over 2.5"
        elif under > 0.55:
            goles = "Under 2.5"
        else:
            goles = "Línea dudosa"

        value_local = calcular_value(prob_local, row["cuota_local"])
        value_visitante = calcular_value(prob_visitante, row["cuota_visitante"])
        value_empate = calcular_value(prob_empate, row["cuota_empate"])

        confianza = calcular_confianza_ajustada(prob, mejor_edge, bonus=confianza_bonus_total)

        if ganador == "Gana local":
            stake_label, stake_num = calcular_stake_kelly(prob_local, row["cuota_local"], factor=stake_factor_total)
        elif ganador == "Gana visitante":
            stake_label, stake_num = calcular_stake_kelly(prob_visitante, row["cuota_visitante"], factor=stake_factor_total)
        elif ganador == "Empate":
            stake_label, stake_num = calcular_stake_kelly(prob_empate, row["cuota_empate"], factor=stake_factor_total)
        else:
            stake_label, stake_num = "❌ NO APOSTAR", 0

        stake_label, stake_num = ajustar_stake_con_cap(stake_num, stake_label, max_stake_cap)

        mercado = generar_mercado(ganador, goles)
        ev_por_unidad = round((prob * row["cuota_local"] - 1), 3) if ganador == "Gana local" else (
            round((prob * row["cuota_visitante"] - 1), 3) if ganador == "Gana visitante" else (
                round((prob * row["cuota_empate"] - 1), 3) if ganador == "Empate" else 0.0
            )
        )

        analisis = (
            f"{row['local']} vs {row['visitante']}\n"
            f"λ: {round(lam_local,2)} vs {round(lam_visitante,2)}\n"
            f"1X2: L {round(prob_local*100,1)}% / E {round(prob_empate*100,1)}% / V {round(prob_visitante*100,1)}%\n"
            f"Edge ganador: {round(mejor_edge*100,1)}%"
        )

        estado, razones = clasificar_partido({
            "confianza": confianza,
            "prob_local": prob_local * 100,
            "prob_visitante": prob_visitante * 100,
            "value_local": value_local,
            "value_visitante": value_visitante
        })

        if trampa and ganador != "No bet":
            ganador = "No bet"
            stake_label, stake_num = "❌ NO APOSTAR", 0
            estado = "NO APOSTAR"
            razones = list(razones) + [f"Trampa detectada: {razon_trampa}"]

        fallback_rescatable = (
            row.get("fuente_cuotas") == "fallback" and
            ganador != "No bet" and
            (
                (
                    float(row.get("prioridad_liga", 0) or 0) >= 85 and
                    confianza >= 54 and
                    mejor_edge_pct >= 7
                ) or
                (
                    confianza >= 58 and
                    mejor_edge_pct >= 9
                )
            )
        )

        if row.get("fuente_cuotas") == "fallback" and confianza < 65 and ganador != "No bet" and not fallback_rescatable:
            ganador = "No bet"
            stake_label, stake_num = "❌ NO APOSTAR", 0
            estado = "NO APOSTAR"
            razones = list(razones) + ["Cuotas fallback con confianza insuficiente"]

        row_result = {
            "fixture_id": row.get("fixture_id"),
            "fecha_partido": row.get("fecha_partido"),
            "hora_partido": row.get("hora_partido"),
            "liga": row.get("liga"),
            "grupo_liga": row.get("grupo_liga"),
            "pais_liga": row.get("pais_liga"),
            "prioridad_liga": row.get("prioridad_liga"),
            "estado_partido": row.get("estado_partido"),
            "local": row.get("local"),
            "visitante": row.get("visitante"),
            "logo_local": row.get("logo_local"),
            "logo_visitante": row.get("logo_visitante"),
            "marcador_local": row.get("marcador_local", 0),
            "marcador_visitante": row.get("marcador_visitante", 0),
            "minuto_partido": row.get("minuto_partido", ""),
            "racha_local": row.get("racha_local"),
            "racha_visitante": row.get("racha_visitante"),
            "ppm_local": row.get("ppm_local"),
            "ppm_visitante": row.get("ppm_visitante"),
            "ppm_local_casa": row.get("ppm_local_casa"),
            "ppm_visitante_fuera": row.get("ppm_visitante_fuera"),
            "record_local": row.get("record_local"),
            "record_visitante": row.get("record_visitante"),
            "goleador_local": row.get("goleador_local"),
            "goles_goleador_local": row.get("goles_goleador_local"),
            "goleador_visitante": row.get("goleador_visitante"),
            "goles_goleador_visitante": row.get("goles_goleador_visitante"),
            "alineacion_confirmada": row.get("alineacion_confirmada", False),
            "once_confirmado_local": row.get("once_confirmado_local", False),
            "once_confirmado_visitante": row.get("once_confirmado_visitante", False),
            "formacion_local": row.get("formacion_local", ""),
            "formacion_visitante": row.get("formacion_visitante", ""),
            "suplentes_local": row.get("suplentes_local", 0),
            "suplentes_visitante": row.get("suplentes_visitante", 0),
            "bajas_local": row.get("bajas_local", 0),
            "bajas_visitante": row.get("bajas_visitante", 0),
            "bajas_suspension_local": row.get("bajas_suspension_local", 0),
            "bajas_suspension_visitante": row.get("bajas_suspension_visitante", 0),
            "bajas_lesion_local": row.get("bajas_lesion_local", 0),
            "bajas_lesion_visitante": row.get("bajas_lesion_visitante", 0),
            "banco_corto_local": row.get("banco_corto_local", False),
            "banco_corto_visitante": row.get("banco_corto_visitante", False),
            "alerta_rotacion_local": row.get("alerta_rotacion_local", False),
            "alerta_rotacion_visitante": row.get("alerta_rotacion_visitante", False),
            "partido": f"{row['local']} vs {row['visitante']}",
            "prob_local": round(prob_local * 100, 1),
            "prob_empate": round(prob_empate * 100, 1),
            "prob_visitante": round(prob_visitante * 100, 1),
            "ganador": ganador,
            "goles": goles,
            "value_local": value_local,
            "value_visitante": value_visitante,
            "value_empate": value_empate,
            "confianza": confianza,
            "stake": stake_label,
            "stake_num": stake_num,
            "mercado": mercado,
            "mercado_historial": mercado,
            "analisis": analisis,
            "estado": estado,
            "cuota_local": row["cuota_local"],
            "cuota_empate": row["cuota_empate"],
            "cuota_visitante": row["cuota_visitante"],
            "fuente_cuotas": row.get("fuente_cuotas", "desconocida"),
            "market_match_score": row.get("market_match_score", 0),
            "bookmaker_count": row.get("bookmaker_count", 0),
            "fuente_partidos": row.get("fuente_partidos", "desconocida"),
            "resultado_real": row.get("resultado_real", None),
            "trampa": trampa,
            "razon_trampa": razon_trampa,
            "min_edge_modelo": round(min_edge_ajustado * 100, 2),
            "mejor_edge": mejor_edge,
            "mejor_edge_pct": mejor_edge_pct,
            "ev_por_unidad": ev_por_unidad,
            "banda_confianza": row.get("banda_confianza"),
            "banda_edge": row.get("banda_edge"),
            "ajustes_modelo": " | ".join(ajustes_contexto["motivos"]) if ajustes_contexto["motivos"] else "Base",
            "aprendizaje_score": round(float(ajustes_contexto.get("aprendizaje_score", 0.0) or 0.0), 2),
            "aprendizaje_favorable": bool(ajustes_contexto.get("aprendizaje_favorable", False)),
            "aprendizaje_desfavorable": bool(ajustes_contexto.get("aprendizaje_desfavorable", False)),
            "razones": ", ".join(razones)
        }
        contexto_comp = _contexto_competitivo({**row.to_dict(), **row_result})
        row_result["contexto_score"] = contexto_comp["score"]
        row_result["contexto_alertas"] = ", ".join(contexto_comp["alertas"]) if contexto_comp["alertas"] else ""
        consenso_analitico = _consenso_analitico({**row.to_dict(), **row_result})
        row_result["consenso_analitico"] = consenso_analitico["nivel"]
        row_result["consenso_score"] = consenso_analitico["score"]
        row_result["consenso_notas"] = ", ".join(consenso_analitico["notas"]) if consenso_analitico["notas"] else ""
        row_result["ai_score"] = calcular_ai_score(row_result)
        row_result["ai_decision"] = etiquetar_ai_decision(row_result["ai_score"])
        row_result["ai_resumen"] = resumir_ai_decision(row_result)
        pred_base, _ = _prediccion_base(row_result)
        row_result["prediccion_simple"] = pred_base if row_result["ganador"] == "No bet" else prediccion_simple(row_result["ganador"])
        row_result["decision_simple"] = decision_simple(row_result)
        row_result["disciplina_simple"] = disciplina_simple(row_result)
        row_result["riesgo_simple"] = riesgo_simple(row_result)
        row_result["contexto_simple"] = contexto_simple({**row.to_dict(), **row_result})
        row_result["razon_simple"] = razon_simple(row_result)

        resultados.append(row_result)

    df_resultados = pd.DataFrame(resultados)
    if df_resultados.empty:
        return df_resultados
    return _promover_mejores_oportunidades(df_resultados)


def generar_combinada_pro(df):

    picks = df[
        (df["confianza"] >= 65) &
        (df["ai_score"] >= 70) &
        (df["ganador"] != "No bet") &
        (df["fuente_cuotas"].isin(["the_odds_api", "espn_odds", "api_football_odds", "api_football_odds_live"])) &
        (~df["ganador"].str.contains("Empate", case=False)) &
        (
            (df["value_local"] == "🟢 VALUE") |
            (df["value_visitante"] == "🟢 VALUE")
        )
    ]

    picks = picks.sort_values(by="confianza", ascending=False)

    combinada = []

    for _, row in picks.iterrows():

        combinada.append({
            "partido": row["partido"],
            "seleccion": row["ganador"],
            "confianza": row["confianza"],
            "stake": row["stake"]
        })

        if len(combinada) == 3:
            break

    return combinada


def filtrar_top_picks(df):

    picks = df[
        (df["confianza"] >= 68) &
        (df["ai_score"] >= 72) &
        (df["ganador"] != "No bet") &
        (df["trampa"] == False) &
        (df["fuente_cuotas"].isin(["the_odds_api", "espn_odds", "api_football_odds", "api_football_odds_live"])) &
        (~df["ganador"].str.contains("Empate", case=False)) &
        (df["estado"] == "APOSTABLE") &
        (
            (df["value_local"] == "🟢 VALUE") |
            (df["value_visitante"] == "🟢 VALUE")
        )
    ]

    picks = picks.sort_values(by="confianza", ascending=False)

    return picks.head(5)
