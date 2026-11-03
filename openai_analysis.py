import json
import os
from typing import Dict, List

import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - graceful fallback when SDK is absent
    OpenAI = None


AI_DECISION_RANK = {"Evitar": 0, "No tocar": 0, "Mirar": 1, "Apostar": 2}
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
        "veredicto_ia": "PENDIENTE",
        "decision_ia": "",
        "prediccion_ia": "",
        "mercado_ia": "",
        "segunda_opinion_ia": "",
        "pick_top1_ia": "",
        "mercado_top1_ia": "",
        "pick_top2_ia": "",
        "mercado_top2_ia": "",
        "pick_top3_ia": "",
        "mercado_top3_ia": "",
        "confianza_ia": None,
        "analisis_ia": "",
        "claves_ia": "",
        "prediccion_modelo": "",
        "decision_modelo": "",
        "pick_modelo": "",
        "mercado_modelo": "",
        "prediccion_final": "",
        "decision_final": "",
        "pick_final": "",
        "mercado_final": "",
        "motivo_decision_final": "",
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
        vista["decision_rank"] = vista["decision_simple"].map({"Apostar": 0, "Mirar": 1, "No tocar": 2, "Evitar": 2}).fillna(9)

    vista["live_rank"] = vista["estado_partido"].isin(["LIVE", "HT", "1H", "2H"]).astype(int)
    vista["prioridad_rank"] = vista["prioridad_liga"].fillna(0)
    vista["confianza_rank"] = vista["confianza"].fillna(0)
    vista["ventaja_rank"] = vista["ventaja_prob_pct"].fillna(0)

    # Las candidatas a dinero real siempre se auditan primero.
    seleccion = vista[
        (
            vista["decision_simple"].eq("Apostar")
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
                "prediccion_partido_modelo": row.get("prediccion_partido", ""),
                "conclusion_prediccion_modelo": row.get("conclusion_prediccion", ""),
                "prediccion_actual": row.get("prediccion_simple", ""),
                "decision_actual": row.get("decision_simple", ""),
                "disciplina_actual": row.get("disciplina_simple", ""),
                "perfil_riesgo": row.get("perfil_riesgo", "equilibrado"),
                "pick_actual": row.get("pick_recomendado", ""),
                "mercado_actual": row.get("mercado_recomendado", ""),
                "prob_pick_actual": row.get("probabilidad_pick", 0),
                "cuota_pick": row.get("cuota_pick"),
                "edge_pick": row.get("edge_pick"),
                "ev_pick": row.get("ev_pick"),
                "riesgo_pick": row.get("riesgo_pick", row.get("riesgo_simple", "")),
                "odds_source_pick": row.get("odds_source_pick", row.get("odds_source", "estimated")),
                "pick_2": row.get("pick_2", ""),
                "mercado_2": row.get("mercado_2", ""),
                "prob_2": row.get("prob_2", 0),
                "pick_3": row.get("pick_3", ""),
                "mercado_3": row.get("mercado_3", ""),
                "prob_3": row.get("prob_3", 0),
                "prob_local_base": round(float(row.get("prob_local_base", 0) or 0), 1),
                "prob_empate_base": round(float(row.get("prob_empate_base", 0) or 0), 1),
                "prob_visitante_base": round(float(row.get("prob_visitante_base", 0) or 0), 1),
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
                "tpi_local": round(float(row.get("tpi_local", 0) or 0), 2),
                "tpi_visitante": round(float(row.get("tpi_visitante", 0) or 0), 2),
                "tpi_diff": round(float(row.get("tpi_diff", 0) or 0), 2),
                "ajuste_jerarquia": row.get("ajuste_jerarquia", ""),
                "motivo_ajuste_1x2": row.get("motivo_ajuste_1x2", ""),
                "auditoria_prediccion_alertas": row.get("auditoria_prediccion_alertas", ""),
                "auditoria_motivo": row.get("auditoria_motivo", ""),
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
        "Eres el ANALISTA PRINCIPAL de fútbol de una aplicación de apuestas. "
        "Tu primera responsabilidad es dar una lectura futbolística independiente: local, empate o visitante. "
        "Usa tu conocimiento general de fútbol, jerarquía competitiva, calidad de plantel, estilo, contexto y los datos entregados. "
        "El modelo matemático es un contraste, no el dueño de la predicción: puedes corregirlo cuando su lectura sea poco creíble. "
        "Después revisa si el mercado recomendado es compatible con tu lectura y con las cuotas reales disponibles. "
        "Debes actuar al mismo tiempo como scout, analista de fútbol y auditor de apuestas. "
        "Debes cuestionar sesgos de localía, sobrepeso de racha reciente, picks incoherentes con la jerarquía futbolística, mercados demasiado cómodos elegidos por inercia, "
        "y contradicciones entre el razonamiento del partido y la apuesta elegida. "
        "Analiza los partidos usando los datos entregados y tu conocimiento general de jerarquía futbolística cuando sea razonable. "
        "No inventes lesiones, cuotas ni alineaciones que no estén en los datos. "
        "Si el motor luce equivocado, corrígelo. Si es razonable, confírmalo. "
        "No uses frases vagas: elige siempre un desenlace 1X2 salvo que de verdad sea imposible separar a los equipos. "
        "Debes hacerte cuatro preguntas antes de responder: "
        "1) ¿Estoy de acuerdo con la lectura 1X2 del modelo? "
        "2) ¿La predicción 1X2 es creíble según TPI, forma y contexto? "
        "3) ¿Los tres mercados principales son coherentes entre sí? "
        "4) ¿Existe un mercado más lógico que no aparece en el top 3 actual? "
        "La IA debe poder decir cosas como: el modelo subestima a España, el modelo sobrevalora la localía, el modelo sobrevalora la racha reciente, el modelo ignora la diferencia de jerarquía. "
        "No puedes inventar lesiones, cuotas, resultados ni EV. Tu rol es definir la lectura futbolística y aprobar, rebajar o rechazar una candidata matemática. "
        "Devuelve exclusivamente un JSON array. "
        "Cada elemento debe tener: fixture_id, veredicto_ia, decision_ia, prediccion_ia, mercado_ia, segunda_opinion_ia, confianza_ia, analisis_ia, claves_ia. "
        "veredicto_ia debe ser exactamente uno de: APROBAR, REBAJAR_A_MIRAR, RECHAZAR. "
        "decision_ia debe ser exactamente uno de: Apostar, Mirar, No tocar. "
        "prediccion_ia debe ser exactamente uno de: Gana el local, Empate, Gana el visitante, Sin lado claro. "
        "mercado_ia debe ser exactamente uno de: Gana local, Gana visitante, Empate, Local o empate, Visitante o empate, Local anota 1+, Visitante anota 1+, Ambos anotan, Over 1.5, Over 2.5, Under 3.5, Under 4.5, No apostar. "
        "segunda_opinion_ia debe ser una sola frase corta en español explicando si confirmas, cuestionas o corriges la lectura del modelo. "
        "pick_top1_ia, pick_top2_ia y pick_top3_ia deben usar exactamente uno de: Gana el local, Gana el visitante, Empate, Local o empate, Visitante o empate, Local anota al menos 1, Visitante anota al menos 1, Ambos anotan, Más de 1.5 goles, Más de 2.5 goles, Menos de 3.5 goles, Menos de 4.5 goles, vacío si no aplica. "
        "mercado_top1_ia, mercado_top2_ia y mercado_top3_ia deben usar exactamente uno de: 1X2, Doble oportunidad, Equipo marca, BTTS, Over 1.5, Over 2.5, Under 3.5, Under 4.5, vacío si no aplica. "
        "confianza_ia debe ser un entero entre 0 y 100. "
        "analisis_ia debe ser una explicación breve en español, máximo 42 palabras, enfocada en por qué confirmas, cuestionas o corriges la predicción y/o la apuesta. "
        "claves_ia debe ser una frase corta con 2 o 3 claves concretas separadas por coma. "
        "Reglas de dinero real: "
        "1) Si el motor favorece a un local débil frente a un visitante de jerarquía claramente superior, debes cuestionarlo o corregirlo. "
        "2) Si el motor subestima a un favorito histórico, debes señalarlo. "
        "3) Si el motor se refugia en un mercado cómodo como Under 4.5 sin ser la mejor apuesta defendible, debes señalarlo. "
        "4) Si no hay cuota real, edge positivo y EV positivo, el veredicto debe ser RECHAZAR. "
        "5) APROBAR solo si la lectura matemática y futbolística es coherente, el riesgo no es alto y el pick tiene sustento. "
        "6) REBAJAR_A_MIRAR si el modelo puede ser razonable pero hay incertidumbre, contexto incompleto o discrepancia relevante. "
        "7) RECHAZAR si hay contradicción importante, cuota no confirmada, EV/edge no validables o riesgo alto. "
        "Datos:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _prediccion_a_lectura(prediccion):
    pred = str(prediccion or "").strip()
    mapping = {
        "Gana el local": "Creo que gana el local",
        "Gana el visitante": "Creo que gana el visitante",
        "Empate": "Empate probable",
        "Sin lado claro": "Partido muy parejo",
    }
    return mapping.get(pred, pred)


def aplicar_filtro_dinero_real(df, auditoria_ia_disponible=False):
    """Aplica la compuerta final: matemática valida + cuota real + auditor IA."""
    vista = _default_ai_columns(df)
    if vista.empty:
        return vista

    for idx, row in vista.iterrows():
        decision_modelo = str(row.get("decision_modelo") or row.get("decision_simple") or "No tocar")
        pick_modelo = str(row.get("pick_modelo") or row.get("pick_recomendado") or "")
        mercado_modelo = str(row.get("mercado_modelo") or row.get("mercado_recomendado") or "")
        fuente = str(row.get("odds_source_pick") or row.get("odds_source") or "estimated").lower()
        try:
            edge = float(row.get("edge_pick") or 0.0)
            ev = float(row.get("ev_pick") or 0.0)
        except (TypeError, ValueError):
            edge, ev = 0.0, 0.0
        riesgo = str(row.get("riesgo_simple") or row.get("riesgo_pick") or "Riesgo alto")
        veredicto = str(row.get("veredicto_ia") or "PENDIENTE").upper()
        cuota = float(row.get("cuota_pick") or 0.0)
        entrada_especulativa = (
            fuente == "real"
            and riesgo == "Riesgo alto"
            and cuota >= 1.80
            and edge >= 6.0
            and ev >= 10.0
        )

        vista.at[idx, "decision_modelo"] = decision_modelo
        vista.at[idx, "pick_modelo"] = pick_modelo
        vista.at[idx, "mercado_modelo"] = mercado_modelo

        motivo = ""
        final = decision_modelo
        if fuente != "real":
            final = "No tocar"
            motivo = "Sin cuota real confirmada, no apto para dinero real."
        elif edge <= 0 or ev <= 0:
            final = "No tocar"
            motivo = "El pick no tiene edge y EV positivos verificables."
        elif riesgo == "Riesgo alto" and not entrada_especulativa:
            final = "No tocar"
            motivo = "Riesgo alto: no apto para dinero real."
        elif decision_modelo != "Apostar" and not entrada_especulativa:
            final = "Mirar" if decision_modelo == "Mirar" else "No tocar"
            motivo = "El filtro matemático no aprobó la apuesta para dinero real."
        elif not auditoria_ia_disponible or veredicto in {"", "PENDIENTE"}:
            final = "Apostar"
            motivo = (
                "Entrada especulativa con valor real: stake bajo por riesgo alto."
                if entrada_especulativa
                else "Modelo matemático aprueba; auditor IA no disponible."
            )
        elif veredicto == "APROBAR":
            final = "Apostar"
            motivo = "Modelo matemático y auditor IA aprueban la apuesta."
        elif veredicto == "REBAJAR_A_MIRAR":
            final = "Mirar"
            motivo = "El auditor IA detectó incertidumbre y rebajó la entrada."
        else:
            final = "No tocar"
            motivo = "El auditor IA rechazó la entrada para dinero real."

        vista.at[idx, "decision_simple"] = final
        vista.at[idx, "decision_final"] = final
        vista.at[idx, "pick_final"] = pick_modelo
        vista.at[idx, "mercado_final"] = mercado_modelo
        vista.at[idx, "motivo_decision_final"] = motivo
        if final == "Apostar":
            vista.at[idx, "disciplina_simple"] = "Apuesta pequeña"
            if entrada_especulativa:
                vista.at[idx, "stake"] = "🟡 MUY BAJO (0.5-1%)"
                vista.at[idx, "stake_num"] = 1
            elif str(row.get("stake") or "").upper() in {"", "❌ NO APOSTAR", "NO APOSTAR"}:
                vista.at[idx, "stake"] = "🟢 BAJO (1-2%)"
                vista.at[idx, "stake_num"] = 2
        elif final == "Mirar":
            vista.at[idx, "disciplina_simple"] = "Solo seguimiento"
        else:
            vista.at[idx, "disciplina_simple"] = "No tocar"

    return vista


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
        veredicto_raw = str(item.get("veredicto_ia", "")).strip().upper()
        veredicto_ia = {
            "RESPALDA": "APROBAR",
            "CUESTIONA": "REBAJAR_A_MIRAR",
            "CORRIGE": "RECHAZAR",
        }.get(veredicto_raw, veredicto_raw if veredicto_raw in {"APROBAR", "REBAJAR_A_MIRAR", "RECHAZAR"} else "RECHAZAR")
        ai_map[fixture_id] = {
            "veredicto_ia": veredicto_ia,
            "decision_ia": str(item.get("decision_ia", "")).strip(),
            "prediccion_ia": str(item.get("prediccion_ia", "")).strip(),
            "mercado_ia": str(item.get("mercado_ia", "")).strip(),
            "segunda_opinion_ia": str(item.get("segunda_opinion_ia", "")).strip(),
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

    for idx, row in vista.iterrows():
        fixture_id = str(row.get("fixture_id", "")).strip()
        if fixture_id not in ai_map:
            continue

        ai_row = ai_map[fixture_id]
        for col, value in ai_row.items():
            vista.at[idx, col] = value

        vista.at[idx, "prediccion_modelo"] = str(row.get("prediccion_partido", "") or "")
        vista.at[idx, "decision_modelo"] = str(row.get("decision_simple", "") or "")
        vista.at[idx, "pick_modelo"] = str(row.get("pick_recomendado", "") or "")
        vista.at[idx, "mercado_modelo"] = str(row.get("mercado_recomendado", "") or "")

        # La IA es la lectura futbolística principal; el modelo conserva el
        # control de probabilidades, cuotas, EV y gestión de riesgo.
        prediccion_ia = str(ai_row.get("prediccion_ia", "") or "").strip()
        if prediccion_ia in {"Gana el local", "Gana el visitante", "Empate", "Sin lado claro"}:
            pred_final = _prediccion_a_lectura(prediccion_ia)
        else:
            pred_final = str(vista.at[idx, "prediccion_partido"] or "")
        vista.at[idx, "prediccion_final"] = pred_final
        vista.at[idx, "decision_final"] = str(vista.at[idx, "decision_simple"] or "No tocar")
        vista.at[idx, "pick_final"] = str(vista.at[idx, "pick_recomendado"] or "")
        vista.at[idx, "mercado_final"] = str(vista.at[idx, "mercado_recomendado"] or "")

    return aplicar_filtro_dinero_real(vista, auditoria_ia_disponible=True)
