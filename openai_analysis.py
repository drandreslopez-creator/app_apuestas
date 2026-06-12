import json
import os
from typing import Dict, List

import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - graceful fallback when SDK is absent
    OpenAI = None


AI_DECISION_RANK = {"Evitar": 0, "Mirar": 1, "Apostar": 2}
AUDIT_MARKET_TO_FIELDS = {
    ("Gana local", "1X2"): ("Gana el local", "1X2", "prob_local"),
    ("Gana visitante", "1X2"): ("Gana el visitante", "1X2", "prob_visitante"),
    ("Empate", "1X2"): ("Empate", "1X2", "prob_empate"),
    ("Local o empate", "Doble oportunidad"): ("Local o empate", "Doble oportunidad", "prob_local_empate"),
    ("Visitante o empate", "Doble oportunidad"): ("Visitante o empate", "Doble oportunidad", "prob_visitante_empate"),
    ("Local anota 1+", "Equipo marca"): ("Local anota al menos 1", "Equipo marca", "prob_local_anota"),
    ("Visitante anota 1+", "Equipo marca"): ("Visitante anota al menos 1", "Equipo marca", "prob_visitante_anota"),
    ("Ambos anotan", "BTTS"): ("Ambos anotan", "BTTS", "prob_btts"),
    ("Over 1.5", "Over 1.5"): ("Más de 1.5 goles", "Over 1.5", "prob_over_15"),
    ("Over 2.5", "Over 2.5"): ("Más de 2.5 goles", "Over 2.5", "prob_over_25"),
    ("Under 3.5", "Under 3.5"): ("Menos de 3.5 goles", "Under 3.5", "prob_under_35"),
    ("Under 4.5", "Under 4.5"): ("Menos de 4.5 goles", "Under 4.5", "prob_under_45"),
}


def _get_openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key

    try:
        import streamlit as st

        secret_key = st.secrets.get("OPENAI_API_KEY")
        if secret_key:
            return secret_key
    except Exception:
        return None

    return None


def openai_analisis_disponible():
    return OpenAI is not None and bool(_get_openai_api_key())


def _default_ai_columns(df):
    vista = df.copy()
    defaults = {
        "veredicto_ia": "",
        "decision_ia": "",
        "prediccion_ia": "",
        "mercado_ia": "",
        "pick_top1_ia": "",
        "mercado_top1_ia": "",
        "pick_top2_ia": "",
        "mercado_top2_ia": "",
        "pick_top3_ia": "",
        "mercado_top3_ia": "",
        "confianza_ia": None,
        "analisis_ia": "",
        "claves_ia": "",
    }
    for col, value in defaults.items():
        if col not in vista.columns:
            vista[col] = value
    return vista


def _seleccionar_para_ia(df, max_partidos=12):
    if df.empty:
        return df.copy()

    vista = df.copy()
    if "decision_rank" not in vista.columns:
        vista["decision_rank"] = vista["decision_simple"].map({"Apostar": 0, "Mirar": 1, "Evitar": 2}).fillna(9)

    vista["live_rank"] = vista["estado_partido"].isin(["LIVE", "HT", "1H", "2H"]).astype(int)
    vista["prioridad_rank"] = vista["prioridad_liga"].fillna(0)
    vista["confianza_rank"] = vista["confianza"].fillna(0)
    vista["ventaja_rank"] = vista["ventaja_prob_pct"].fillna(0)

    seleccion = vista[
        (
            vista["decision_simple"].isin(["Apostar", "Mirar"])
            | (vista["prioridad_rank"] >= 90)
            | (vista["ventaja_rank"] >= 12)
        )
    ].copy()

    if seleccion.empty:
        seleccion = vista.copy()

    seleccion = seleccion.sort_values(
        by=["decision_rank", "live_rank", "prioridad_rank", "confianza_rank", "ventaja_rank"],
        ascending=[True, False, False, False, False],
    )
    return seleccion.head(max_partidos).copy()


def _extraer_json(texto):
    texto = (texto or "").strip()
    if not texto:
        return []

    if texto.startswith("```"):
        partes = texto.split("```")
        for parte in partes:
            parte = parte.strip()
            if parte.startswith("[") or parte.startswith("{"):
                texto = parte
                break
            if "\n" in parte:
                _, posible = parte.split("\n", 1)
                posible = posible.strip()
                if posible.startswith("[") or posible.startswith("{"):
                    texto = posible
                    break

    inicio_array = texto.find("[")
    fin_array = texto.rfind("]")
    if inicio_array != -1 and fin_array != -1 and fin_array > inicio_array:
        texto = texto[inicio_array:fin_array + 1]

    try:
        data = json.loads(texto)
        if isinstance(data, list):
            return data
    except Exception:
        return []

    return []


def _build_payload_rows(df):
    payload = []
    for _, row in df.iterrows():
        payload.append(
            {
                "fixture_id": str(row.get("fixture_id", "")),
                "partido": row.get("partido", ""),
                "liga": row.get("liga", ""),
                "estado": row.get("estado_partido", ""),
                "prediccion_actual": row.get("prediccion_simple", ""),
                "decision_actual": row.get("decision_simple", ""),
                "disciplina_actual": row.get("disciplina_simple", ""),
                "perfil_riesgo": row.get("perfil_riesgo", "equilibrado"),
                "pick_actual": row.get("pick_recomendado", ""),
                "mercado_actual": row.get("mercado_recomendado", ""),
                "prob_pick_actual": row.get("probabilidad_pick", 0),
                "pick_2": row.get("pick_2", ""),
                "mercado_2": row.get("mercado_2", ""),
                "prob_2": row.get("prob_2", 0),
                "pick_3": row.get("pick_3", ""),
                "mercado_3": row.get("mercado_3", ""),
                "prob_3": row.get("prob_3", 0),
                "prob_local": round(float(row.get("prob_local", 0) or 0), 1),
                "prob_empate": round(float(row.get("prob_empate", 0) or 0), 1),
                "prob_visitante": round(float(row.get("prob_visitante", 0) or 0), 1),
                "prob_local_empate": round(float(row.get("prob_local_empate", 0) or 0), 1),
                "prob_visitante_empate": round(float(row.get("prob_visitante_empate", 0) or 0), 1),
                "prob_local_anota": round(float(row.get("prob_local_anota", 0) or 0), 1),
                "prob_visitante_anota": round(float(row.get("prob_visitante_anota", 0) or 0), 1),
                "prob_over_15": round(float(row.get("prob_over_15", 0) or 0), 1),
                "prob_over_25": round(float(row.get("prob_over_25", 0) or 0), 1),
                "prob_under_35": round(float(row.get("prob_under_35", 0) or 0), 1),
                "prob_under_45": round(float(row.get("prob_under_45", 0) or 0), 1),
                "prob_btts": round(float(row.get("prob_btts", 0) or 0), 1),
                "cuota_local": round(float(row.get("cuota_local", 0) or 0), 2),
                "cuota_empate": round(float(row.get("cuota_empate", 0) or 0), 2),
                "cuota_visitante": round(float(row.get("cuota_visitante", 0) or 0), 2),
                "mejor_edge_pct": round(float(row.get("mejor_edge_pct", 0) or 0), 2),
                "ventaja_prob_pct": round(float(row.get("ventaja_prob_pct", 0) or 0), 2),
                "confianza": int(float(row.get("confianza", 0) or 0)),
                "consenso": row.get("consenso_analitico", ""),
                "consenso_score": round(float(row.get("consenso_score", 0) or 0), 2),
                "contexto_score": round(float(row.get("contexto_score", 0) or 0), 2),
                "contexto_alertas": row.get("contexto_alertas", ""),
                "fuente_cuotas": row.get("fuente_cuotas", ""),
                "racha_local": row.get("racha_local", ""),
                "racha_visitante": row.get("racha_visitante", ""),
                "ppm_local": row.get("ppm_local", 0),
                "ppm_visitante": row.get("ppm_visitante", 0),
                "ppm_local_casa": row.get("ppm_local_casa", 0),
                "ppm_visitante_fuera": row.get("ppm_visitante_fuera", 0),
                "record_local": row.get("record_local", ""),
                "record_visitante": row.get("record_visitante", ""),
                "goleador_local": row.get("goleador_local", ""),
                "goles_goleador_local": row.get("goles_goleador_local", 0),
                "goleador_visitante": row.get("goleador_visitante", ""),
                "goles_goleador_visitante": row.get("goles_goleador_visitante", 0),
                "bajas_local": row.get("bajas_local", 0),
                "bajas_visitante": row.get("bajas_visitante", 0),
                "alerta_rotacion_local": bool(row.get("alerta_rotacion_local", False)),
                "alerta_rotacion_visitante": bool(row.get("alerta_rotacion_visitante", False)),
                "alineacion_confirmada": bool(row.get("alineacion_confirmada", False)),
                "contexto": row.get("contexto_simple", ""),
                "conclusion_actual": row.get("razon_simple", ""),
            }
        )
    return payload


def _prompt_analisis(payload):
    return (
        "Eres un AUDITOR experto de un modelo de apuestas deportivas. "
        "Tu trabajo NO es repetir los datos ni embellecer el texto. Tu trabajo es buscar fallas de lógica del motor principal. "
        "Debes cuestionar sesgos de localía, picks incoherentes con la jerarquía futbolística, mercados demasiado cómodos elegidos por inercia, "
        "y contradicciones entre el razonamiento y el mercado recomendado. "
        "Analiza los partidos usando los datos entregados y tu conocimiento general de jerarquía futbolística cuando sea razonable. "
        "No inventes lesiones, cuotas ni alineaciones que no estén en los datos. "
        "Si el motor luce equivocado, debes corregirlo. Si el motor está razonable, debes respaldarlo. "
        "No uses frases vagas. Habla como segundo analista crítico. "
        "Debes hacerte dos preguntas antes de responder: "
        "1) ¿Los tres mercados principales son coherentes entre sí? "
        "2) ¿Existe un mercado más lógico que no aparece en el top 3 actual? "
        "Si detectas incoherencia, puedes reordenar o sustituir el top 3 final. "
        "Devuelve exclusivamente un JSON array. "
        "Cada elemento debe tener: fixture_id, veredicto_ia, decision_ia, prediccion_ia, mercado_ia, pick_top1_ia, mercado_top1_ia, pick_top2_ia, mercado_top2_ia, pick_top3_ia, mercado_top3_ia, confianza_ia, analisis_ia, claves_ia. "
        "veredicto_ia debe ser exactamente uno de: RESPALDA, CUESTIONA, CORRIGE. "
        "decision_ia debe ser exactamente uno de: Apostar, Mirar, Evitar. "
        "prediccion_ia debe ser exactamente uno de: Gana el local, Empate, Gana el visitante, Sin lado claro. "
        "mercado_ia debe ser exactamente uno de: Gana local, Gana visitante, Empate, Local o empate, Visitante o empate, Local anota 1+, Visitante anota 1+, Ambos anotan, Over 1.5, Over 2.5, Under 3.5, Under 4.5, No apostar. "
        "pick_top1_ia, pick_top2_ia y pick_top3_ia deben usar exactamente uno de: Gana el local, Gana el visitante, Empate, Local o empate, Visitante o empate, Local anota al menos 1, Visitante anota al menos 1, Ambos anotan, Más de 1.5 goles, Más de 2.5 goles, Menos de 3.5 goles, Menos de 4.5 goles, vacío si no aplica. "
        "mercado_top1_ia, mercado_top2_ia y mercado_top3_ia deben usar exactamente uno de: 1X2, Doble oportunidad, Equipo marca, BTTS, Over 1.5, Over 2.5, Under 3.5, Under 4.5, vacío si no aplica. "
        "confianza_ia debe ser un entero entre 0 y 100. "
        "analisis_ia debe ser una explicación breve en español, máximo 42 palabras, enfocada en la falla o validación del motor. "
        "claves_ia debe ser una frase corta con 2 o 3 claves concretas separadas por coma. "
        "Reglas clave: "
        "1) Si el motor favorece a un local débil frente a un visitante de jerarquía claramente superior, debes cuestionarlo o corregirlo. "
        "2) Si el motor se refugia en un mercado cómodo como Under 4.5 sin ser la mejor apuesta defendible, debes señalarlo. "
        "3) Si el razonamiento sugiere una cosa y el mercado elegido otra, debes corregir el mercado. "
        "4) No estás obligado a coincidir con el motor. "
        "5) Solo respalda cuando la lógica del motor aguanta el escrutinio. "
        "Datos:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def analizar_partidos_con_openai(df, max_partidos=12):
    vista = _default_ai_columns(df)

    if vista.empty or OpenAI is None:
        return vista

    api_key = _get_openai_api_key()
    if not api_key:
        return vista

    subset = _seleccionar_para_ia(vista, max_partidos=max_partidos)
    if subset.empty:
        return vista

    payload = _build_payload_rows(subset)
    client = OpenAI(api_key=api_key)
    model_name = os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-5.5")

    try:
        response = client.responses.create(
            model=model_name,
            store=False,
            input=_prompt_analisis(payload),
        )
        data = _extraer_json(getattr(response, "output_text", ""))
    except Exception:
        return vista

    if not data:
        return vista

    ai_map: Dict[str, Dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        fixture_id = str(item.get("fixture_id", "")).strip()
        if not fixture_id:
            continue
        ai_map[fixture_id] = {
            "veredicto_ia": str(item.get("veredicto_ia", "")).strip(),
            "decision_ia": str(item.get("decision_ia", "")).strip(),
            "prediccion_ia": str(item.get("prediccion_ia", "")).strip(),
            "mercado_ia": str(item.get("mercado_ia", "")).strip(),
            "pick_top1_ia": str(item.get("pick_top1_ia", "")).strip(),
            "mercado_top1_ia": str(item.get("mercado_top1_ia", "")).strip(),
            "pick_top2_ia": str(item.get("pick_top2_ia", "")).strip(),
            "mercado_top2_ia": str(item.get("mercado_top2_ia", "")).strip(),
            "pick_top3_ia": str(item.get("pick_top3_ia", "")).strip(),
            "mercado_top3_ia": str(item.get("mercado_top3_ia", "")).strip(),
            "confianza_ia": item.get("confianza_ia"),
            "analisis_ia": str(item.get("analisis_ia", "")).strip(),
            "claves_ia": str(item.get("claves_ia", "")).strip(),
        }

    if not ai_map:
        return vista

    def _aplicar_top3_auditor(idx):
        for pos in range(1, 4):
            audit_pick = str(vista.at[idx, f"pick_top{pos}_ia"] or "").strip()
            audit_market = str(vista.at[idx, f"mercado_top{pos}_ia"] or "").strip()
            if not audit_pick or not audit_market:
                continue
            mapping = AUDIT_MARKET_TO_FIELDS.get((audit_pick.replace("Local anota al menos 1", "Local anota 1+").replace("Visitante anota al menos 1", "Visitante anota 1+"), audit_market))
            if not mapping:
                reverse = {
                    "Gana el local": ("Gana local", "1X2"),
                    "Gana el visitante": ("Gana visitante", "1X2"),
                    "Empate": ("Empate", "1X2"),
                    "Local o empate": ("Local o empate", "Doble oportunidad"),
                    "Visitante o empate": ("Visitante o empate", "Doble oportunidad"),
                    "Local anota al menos 1": ("Local anota 1+", "Equipo marca"),
                    "Visitante anota al menos 1": ("Visitante anota 1+", "Equipo marca"),
                    "Ambos anotan": ("Ambos anotan", "BTTS"),
                    "Más de 1.5 goles": ("Over 1.5", "Over 1.5"),
                    "Más de 2.5 goles": ("Over 2.5", "Over 2.5"),
                    "Menos de 3.5 goles": ("Under 3.5", "Under 3.5"),
                    "Menos de 4.5 goles": ("Under 4.5", "Under 4.5"),
                }
                pair = reverse.get(audit_pick)
                mapping = AUDIT_MARKET_TO_FIELDS.get(pair) if pair else None
            if not mapping:
                continue
            canonical_pick, canonical_market, prob_field = mapping
            vista.at[idx, f"pick_{pos}"] = canonical_pick
            vista.at[idx, f"mercado_{pos}"] = canonical_market
            vista.at[idx, f"prob_{pos}"] = float(vista.at[idx, prob_field] or 0)
            vista.at[idx, f"confianza_{pos}"] = ""
            vista.at[idx, f"riesgo_{pos}"] = ""

    for idx, row in vista.iterrows():
        fixture_id = str(row.get("fixture_id", "")).strip()
        if fixture_id not in ai_map:
            continue

        ai_row = ai_map[fixture_id]
        for col, value in ai_row.items():
            vista.at[idx, col] = value

        decision_actual = str(vista.at[idx, "decision_simple"] or "Evitar")
        disciplina_actual = str(vista.at[idx, "disciplina_simple"] or "No tocar")
        veredicto_ia = ai_row.get("veredicto_ia", "")
        decision_ia = ai_row.get("decision_ia", "")
        pred_actual = str(vista.at[idx, "prediccion_simple"] or "")
        pred_ia = ai_row.get("prediccion_ia", "")
        mercado_ia = str(ai_row.get("mercado_ia", "") or "").strip()
        confianza_ia = float(ai_row.get("confianza_ia") or 0)
        ventaja_prob = float(vista.at[idx, "ventaja_prob_pct"] or 0)
        perfil_riesgo = str(vista.at[idx, "perfil_riesgo"] or "equilibrado").lower()
        trampa = bool(vista.at[idx, "trampa"]) if "trampa" in vista.columns else False

        if trampa:
            continue

        misma_lectura = pred_actual == pred_ia and pred_ia in {"Gana el local", "Gana el visitante", "Empate"}
        rank_actual = AI_DECISION_RANK.get(decision_actual, 0)
        rank_ia = AI_DECISION_RANK.get(decision_ia, 0)

        if veredicto_ia in {"CORRIGE", "CUESTIONA"}:
            _aplicar_top3_auditor(idx)
            if str(vista.at[idx, "pick_1"] or "").strip():
                vista.at[idx, "pick_recomendado"] = vista.at[idx, "pick_1"]
                vista.at[idx, "mercado_recomendado"] = vista.at[idx, "mercado_1"]
                vista.at[idx, "probabilidad_pick"] = vista.at[idx, "prob_1"]

        if veredicto_ia == "CORRIGE" and confianza_ia >= 70:
            if pred_ia and pred_ia != "Sin lado claro":
                vista.at[idx, "prediccion_simple"] = pred_ia
            if mercado_ia and mercado_ia != "No apostar":
                vista.at[idx, "mercado_recomendado"] = mercado_ia
                vista.at[idx, "pick_recomendado"] = mercado_ia
            vista.at[idx, "decision_simple"] = decision_ia or decision_actual
            if decision_ia == "Apostar":
                vista.at[idx, "disciplina_simple"] = "Apuesta pequeña" if confianza_ia < 80 else "Apuesta fuerte"
                vista.at[idx, "riesgo_simple"] = "Riesgo medio" if confianza_ia < 80 else "Riesgo controlado"
            elif decision_ia == "Mirar":
                vista.at[idx, "disciplina_simple"] = "Entrada pequeña" if perfil_riesgo == "agresivo" else "Solo seguimiento"
                vista.at[idx, "riesgo_simple"] = "Riesgo medio"
            else:
                vista.at[idx, "disciplina_simple"] = "No tocar"
                vista.at[idx, "riesgo_simple"] = "Riesgo alto"
        elif veredicto_ia == "CUESTIONA" and confianza_ia >= 66:
            if decision_actual == "Apostar":
                vista.at[idx, "decision_simple"] = "Mirar"
                vista.at[idx, "disciplina_simple"] = "Solo seguimiento"
                vista.at[idx, "riesgo_simple"] = "Riesgo medio"
            elif decision_actual == "Mirar" and rank_ia == 0:
                vista.at[idx, "disciplina_simple"] = "No tocar"
                vista.at[idx, "riesgo_simple"] = "Riesgo alto"
        elif (
            veredicto_ia == "RESPALDA" and
            decision_ia == "Apostar" and
            confianza_ia >= (58 if perfil_riesgo == "agresivo" else 62) and
            (misma_lectura or perfil_riesgo == "agresivo")
        ):
            vista.at[idx, "decision_simple"] = "Apostar"
            vista.at[idx, "disciplina_simple"] = "Apuesta pequeña" if confianza_ia < 80 else "Apuesta fuerte"
            vista.at[idx, "riesgo_simple"] = "Riesgo medio" if confianza_ia < 80 else "Riesgo controlado"

        analisis_ia = ai_row.get("analisis_ia", "")
        claves_ia = ai_row.get("claves_ia", "")
        if analisis_ia:
            vista.at[idx, "analisis_ia"] = analisis_ia
        if claves_ia:
            vista.at[idx, "claves_ia"] = claves_ia

    return vista
