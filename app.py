import streamlit as st
from datetime import datetime, timedelta

from tracker import (
    calcular_roi,
    actualizar_resultados,
    calcular_calibracion_modelo,
    guardar_resultado,
    resumen_historial,
)
from model import analizar_partidos
from api_data import get_matches_api, get_last_api_status
from openai_analysis import analizar_partidos_con_openai, openai_analisis_disponible

VENTANA_PARTIDOS = "hoy y próximos 2 días"
DECISION_ORDEN = {"Apostar": 0, "Mirar": 1, "No tocar": 2, "Evitar": 2}
LIVE_REFRESH_INTERVAL = "45s"
AUTO_HISTORY_SYNC_MINUTES = 20
COLOR_TEXT_MAIN = "var(--text-color)"
COLOR_TEXT_MUTED = "color-mix(in srgb, var(--text-color) 64%, transparent)"
COLOR_TEXT_SOFT = "color-mix(in srgb, var(--text-color) 52%, transparent)"
COLOR_BORDER = "color-mix(in srgb, var(--text-color) 12%, transparent)"
COLOR_CARD_BG = "color-mix(in srgb, var(--secondary-background-color) 94%, var(--background-color) 6%)"


@st.cache_data(ttl=120, show_spinner=False)
def cargar_partidos():
    return get_matches_api(limit=80)


@st.cache_data(ttl=120, show_spinner=False)
def cargar_resultados(df, calibracion, perfil_riesgo):
    return analizar_partidos(df.copy(), calibracion=calibracion, perfil_riesgo=perfil_riesgo)


@st.cache_data(ttl=300, show_spinner=False)
def cargar_resultados_openai(df):
    return analizar_partidos_con_openai(df.copy(), max_partidos=12)


@st.cache_data(ttl=120, show_spinner=False)
def cargar_resumen_historial():
    return resumen_historial()


@st.cache_data(ttl=300, show_spinner=False)
def cargar_calibracion_modelo():
    return calcular_calibracion_modelo()


def sincronizar_historial_automatico():
    ahora = datetime.now()
    ultima = st.session_state.get("historial_auto_sync_at")
    if ultima and (ahora - ultima) < timedelta(minutes=AUTO_HISTORY_SYNC_MINUTES):
        return False

    st.session_state["historial_auto_sync_at"] = ahora
    try:
        actualizado = actualizar_resultados()
    except Exception:
        return False

    if actualizado:
        cargar_resumen_historial.clear()
        cargar_calibracion_modelo.clear()
    return actualizado


def badge_decision(decision):
    colores = {
        "Apostar": ("#133a1b", "#8ef0a7"),
        "Mirar": ("#3a3213", "#f3d97a"),
        "Evitar": ("#3a1616", "#ff9898"),
        "No tocar": ("#3a1616", "#ff9898"),
    }
    bg, fg = colores.get(decision, ("#1c2430", "#d7e3f4"))
    return f"<span style='background:{bg};color:{fg};padding:4px 10px;border-radius:999px;font-size:12px;font-weight:700;'>{decision}</span>"


def badge_estado(estado):
    if estado in {"LIVE", "HT", "1H", "2H"}:
        return (
            "<span style='display:inline-flex;align-items:center;gap:6px;'>"
            "<span style='width:8px;height:8px;border-radius:999px;background:#ef4444;"
            "display:inline-block;box-shadow:0 0 0 2px rgba(239,68,68,0.15);"
            "animation:livePulse 1.6s ease-out infinite;'></span>"
            f"<span style='color:#ff7b7b;font-size:12px;font-weight:700;'>{estado}</span>"
            "</span>"
        )
    return f"<span style='color:{COLOR_TEXT_MUTED};font-size:12px;font-weight:600;'>{estado}</span>"


def preparar_resultados_ui(df):
    vista = df.copy()
    defaults = {
        "liga": "Sin liga",
        "grupo_liga": "Otras ligas",
        "partido": "",
        "hora_partido": "",
        "estado_partido": "NS",
        "prediccion_simple": "Partido para mirar",
        "decision_simple": "Mirar",
        "disciplina_simple": "Solo seguimiento",
        "riesgo_simple": "Riesgo medio",
        "contexto_simple": "Sin contexto adicional",
        "razon_simple": "",
        "perfil_riesgo": "equilibrado",
        "pick_recomendado": "",
        "mercado_recomendado": "",
        "probabilidad_pick": None,
        "confianza_pick": "",
        "claves_pick": "",
        "cuota_pick": None,
        "edge_pick": 0.0,
        "ev_pick": 0.0,
        "score_pick": 0.0,
        "score_final": 0.0,
        "justificacion_ranking": "",
        "mercado_observado": "",
        "pick_observado": "",
        "motivo_descarte": "",
        "cuota_1": None,
        "edge_1": 0.0,
        "ev_1": 0.0,
        "score_pick_1": 0.0,
        "score_final_1": 0.0,
        "riesgo_pick": "",
        "valor_pick": 0.0,
        "pick_1": "",
        "mercado_1": "",
        "prob_1": None,
        "confianza_1": "",
        "riesgo_1": "",
        "valor_1": 0.0,
        "claves_1": "",
        "pick_2": "",
        "mercado_2": "",
        "prob_2": None,
        "confianza_2": "",
        "riesgo_2": "",
        "valor_2": 0.0,
        "cuota_2": None,
        "edge_2": 0.0,
        "ev_2": 0.0,
        "score_pick_2": 0.0,
        "score_final_2": 0.0,
        "claves_2": "",
        "pick_3": "",
        "mercado_3": "",
        "prob_3": None,
        "confianza_3": "",
        "riesgo_3": "",
        "valor_3": 0.0,
        "cuota_3": None,
        "edge_3": 0.0,
        "ev_3": 0.0,
        "score_pick_3": 0.0,
        "score_final_3": 0.0,
        "claves_3": "",
        "auditoria_inconsistencias": 0,
        "auditoria_partido_afectado": False,
        "auditoria_mercados_corregidos": "",
        "auditoria_detalle": "",
        "decision_ia": "",
        "prediccion_ia": "",
        "veredicto_ia": "",
        "mercado_ia": "",
        "confianza_ia": None,
        "analisis_ia": "",
        "claves_ia": "",
        "prioridad_liga": 0,
    }

    for col, value in defaults.items():
        if col not in vista.columns:
            vista[col] = value
        else:
            if value is None:
                vista[col] = vista[col].where(vista[col].notna(), None)
            else:
                vista[col] = vista[col].fillna(value)

    vista["decision_rank"] = vista["decision_simple"].map(DECISION_ORDEN).fillna(9).astype(int)
    vista["prioridad_liga"] = vista["prioridad_liga"].fillna(0)
    return vista


def resumen_fuente(api_status):
    source = api_status.get("source")
    if source == "cache_local":
        return "Se está usando la última carga guardada localmente."
    if source == "espn_scoreboard":
        return "Se cargaron partidos usando ESPN como respaldo."
    if api_status.get("message"):
        return api_status["message"]
    return "Partidos cargados correctamente."


def seleccionar_partidos_destacados(df, max_items=3):
    if df.empty:
        return df

    vista = df.copy()
    vista["en_vivo_rank"] = vista["estado_partido"].isin(["LIVE", "HT", "1H", "2H"]).astype(int)
    vista["decision_rank"] = vista["decision_simple"].map(DECISION_ORDEN).fillna(9).astype(int)
    vista["confianza_rank"] = vista["confianza"].fillna(0)
    vista["prioridad_rank"] = vista["prioridad_liga"].fillna(0)

    destacados = vista.sort_values(
        by=["decision_rank", "en_vivo_rank", "prioridad_rank", "confianza_rank", "hora_partido"],
        ascending=[True, False, False, False, True],
    )
    destacados = destacados[destacados["decision_simple"].isin(["Apostar", "Mirar"])]

    if destacados.empty:
        destacados = vista.sort_values(
            by=["en_vivo_rank", "prioridad_rank", "confianza_rank", "hora_partido"],
            ascending=[False, False, False, True],
        )

    if max_items is None or max_items <= 0:
        return destacados.copy()
    return destacados.head(max_items).copy()


def seleccionar_partidos_en_vivo(df, max_items=6):
    if df.empty:
        return df

    vista = df.copy()
    vista = vista[vista["estado_partido"].isin(["LIVE", "HT", "1H", "2H"])].copy()
    if vista.empty:
        return vista

    vista["decision_rank"] = vista["decision_simple"].map(DECISION_ORDEN).fillna(9).astype(int)
    vista["confianza_rank"] = vista["confianza"].fillna(0)
    vista["prioridad_rank"] = vista["prioridad_liga"].fillna(0)
    vista = vista.sort_values(
        by=["decision_rank", "prioridad_rank", "confianza_rank", "hora_partido"],
        ascending=[True, False, False, True],
    )
    return vista.head(max_items).copy()


def render_grid_tarjetas(df, mostrar_disciplina=False, cards_per_row=3):
    if df.empty:
        return

    cards_per_row = max(1, int(cards_per_row))
    rows = [df.iloc[i:i + cards_per_row] for i in range(0, len(df), cards_per_row)]
    for bloque in rows:
        columnas = st.columns(cards_per_row)
        for idx, (_, row) in enumerate(bloque.iterrows()):
            with columnas[idx]:
                render_tarjeta_compacta(row, mostrar_disciplina=mostrar_disciplina)
        st.markdown("")


def render_tarjeta_compacta(row, mostrar_disciplina=False):
    local = row.get("local", "Local")
    visitante = row.get("visitante", "Visitante")
    liga_top = row.get("grupo_liga") or row.get("liga")
    hora_top = row.get("hora_partido", "")
    estado_top = row.get("estado_partido", "NS")
    decision_top = row.get("decision_simple", "Mirar")
    pred_top = row.get("prediccion_simple", "Partido para mirar")
    pick_top = row.get("pick_recomendado") or pred_top
    mercado_top = row.get("mercado_recomendado") or "1X2"
    disciplina_top = row.get("disciplina_simple", "Solo seguimiento")
    riesgo_top = row.get("riesgo_simple", "Riesgo medio")
    consenso_top = row.get("consenso_analitico", "")
    analisis_ia = str(row.get("analisis_ia", "") or "").strip()
    claves_ia = str(row.get("claves_ia", "") or "").strip()
    logo_local = row.get("logo_local") or ""
    logo_visitante = row.get("logo_visitante") or ""
    marcador_local = row.get("marcador_local", 0)
    marcador_visitante = row.get("marcador_visitante", 0)
    minuto_partido = row.get("minuto_partido", "")
    estado_actual = str(estado_top or "NS").upper()
    marcador_html = f"{marcador_local}-{marcador_visitante}" if estado_actual in {"LIVE", "HT", "1H", "2H"} else "vs"

    st.markdown(
        f"<div style='border:1px solid {COLOR_BORDER};border-radius:12px;padding:10px 12px;background:{COLOR_CARD_BG};'>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:10px;color:{COLOR_TEXT_SOFT};height:16px;'>{liga_top}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:minmax(0,1fr) 42px minmax(0,1fr);align-items:center;column-gap:8px;margin-top:4px;">
            <div style="display:flex;align-items:center;justify-content:flex-end;gap:8px;min-width:0;">
                {f'<img src="{logo_local}" style="width:22px;height:22px;object-fit:contain;" />' if logo_local else ''}
                <div style="font-size:13px;font-weight:800;color:{COLOR_TEXT_MAIN};line-height:1.2;text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{local}</div>
            </div>
            <div style="font-size:11px;color:{COLOR_TEXT_MUTED};font-weight:700;text-align:center;display:flex;align-items:center;justify-content:center;">{marcador_html}</div>
            <div style="display:flex;align-items:center;justify-content:flex-start;gap:8px;min-width:0;">
                {f'<img src="{logo_visitante}" style="width:22px;height:22px;object-fit:contain;" />' if logo_visitante else ''}
                <div style="font-size:13px;font-weight:800;color:{COLOR_TEXT_MAIN};line-height:1.2;text-align:left;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{visitante}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div style='font-size:11px;color:{COLOR_TEXT_SOFT};margin-top:8px;height:18px;text-align:center;'>{hora_top} | {badge_estado(estado_top)}</div>",
        unsafe_allow_html=True,
    )
    if estado_actual in {"LIVE", "HT", "1H", "2H"} and minuto_partido:
        st.markdown(
            f"<div style='font-size:11px;color:{COLOR_TEXT_MUTED};text-align:center;font-weight:700;height:16px;margin-top:2px;'>{minuto_partido}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    st.markdown(
        (
            "<div style='display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);"
            "align-items:start;column-gap:8px;margin-top:8px;'>"
            "<div></div>"
            "<div style='min-width:120px;display:flex;flex-direction:column;align-items:center;"
            "justify-content:center;text-align:center;'>"
            f"{badge_decision(decision_top)}"
            f"<div style='font-size:11px;color:{COLOR_TEXT_MAIN};font-weight:700;text-align:center;margin-top:4px;'>{pick_top}</div>"
            f"<div style='font-size:10px;color:{COLOR_TEXT_MUTED};font-weight:700;text-align:center;margin-top:2px;'>{mercado_top}</div>"
            + (
                f"<div style='font-size:10px;color:{COLOR_TEXT_MUTED};font-weight:700;text-align:center;margin-top:3px;'>Consenso {consenso_top}</div>"
                if consenso_top else
                ""
            )
            + (
                f"<div style='font-size:10px;color:{COLOR_TEXT_SOFT};font-weight:600;text-align:center;margin-top:3px;'>{disciplina_top} · {riesgo_top}</div>"
                if mostrar_disciplina else
                ""
            )
            + (
                f"<div style='font-size:10px;color:{COLOR_TEXT_MUTED};font-weight:600;text-align:center;margin-top:3px;'>Socio IA: {analisis_ia}</div>"
                if analisis_ia else
                ""
            )
            + (
                f"<div style='font-size:10px;color:{COLOR_TEXT_SOFT};font-weight:600;text-align:center;margin-top:3px;'>Claves: {claves_ia}</div>"
                if claves_ia else
                ""
            )
            + "</div><div></div></div>"
        ),
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_partido(row):
    def _is_missing(value):
        try:
            return value in ("", None) or bool(value != value)
        except Exception:
            return value in ("", None)

    def _fmt_pct(value):
        return "—" if _is_missing(value) else f"{value}%"

    def _fmt_num(value):
        return "—" if _is_missing(value) else f"{value}"

    local = row.get("local")
    visitante = row.get("visitante")
    if (not local or not visitante) and row.get("partido"):
        partes = str(row.get("partido")).split(" vs ", 1)
        if len(partes) == 2:
            local = local or partes[0]
            visitante = visitante or partes[1]
    local = local or "Equipo local"
    visitante = visitante or "Equipo visitante"
    marcador_local = row.get("marcador_local", 0)
    marcador_visitante = row.get("marcador_visitante", 0)
    minuto_partido = row.get("minuto_partido", "")
    estado_actual = str(row.get("estado_partido", "NS") or "NS").upper()
    mostrar_marcador = estado_actual in {"LIVE", "HT", "1H", "2H"} or (int(marcador_local or 0) + int(marcador_visitante or 0) > 0)
    consenso = str(row.get("consenso_analitico", "") or "").strip()
    consenso_score = row.get("consenso_score", "")
    consenso_notas = str(row.get("consenso_notas", "") or "").strip()
    analisis_ia = str(row.get("analisis_ia", "") or "").strip()
    veredicto_ia = str(row.get("veredicto_ia", "") or "").strip()
    decision_ia = str(row.get("decision_ia", "") or "").strip()
    mercado_ia = str(row.get("mercado_ia", "") or "").strip()
    claves_ia = str(row.get("claves_ia", "") or "").strip()
    pick_recomendado = row.get("pick_recomendado") or row.get("prediccion_simple", "Partido para mirar")
    mercado_recomendado = row.get("mercado_recomendado") or "1X2"
    probabilidad_pick = row.get("probabilidad_pick")
    confianza_pick = row.get("confianza_pick", "")
    claves_pick = row.get("claves_pick", "")
    cuota_pick = row.get("cuota_pick")
    edge_pick = row.get("edge_pick", 0.0)
    ev_pick = row.get("ev_pick", 0.0)
    score_final = row.get("score_final", 0.0)
    justificacion_ranking = str(row.get("justificacion_ranking", "") or "").strip()
    mercado_observado = str(row.get("mercado_observado", "") or "").strip()
    pick_observado = str(row.get("pick_observado", "") or "").strip()
    motivo_descarte = str(row.get("motivo_descarte", "") or "").strip()
    sin_valor_real = str(pick_recomendado).strip().lower() == "no hay picks con valor real"
    top_picks = []
    for idx in range(1, 4):
        pick = str(row.get(f"pick_{idx}", "") or "").strip()
        mercado = str(row.get(f"mercado_{idx}", "") or "").strip()
        prob = row.get(f"prob_{idx}")
        if pick and mercado and prob not in ("", None):
            top_picks.append(
                {
                    "label": ["Mejor pick", "Segundo pick", "Tercer pick"][idx - 1],
                    "pick": pick,
                    "mercado": mercado,
                    "prob": prob,
                    "confianza": row.get(f"confianza_{idx}", ""),
                    "riesgo": row.get(f"riesgo_{idx}", ""),
                    "cuota": row.get(f"cuota_{idx}"),
                    "edge": row.get(f"edge_{idx}", 0.0),
                    "ev": row.get(f"ev_{idx}", 0.0),
                    "score_final": row.get(f"score_final_{idx}", 0.0),
                }
            )

    st.markdown(f"<div style='padding:10px 0;border-bottom:1px solid {COLOR_BORDER};'>", unsafe_allow_html=True)

    top1, top2, top3 = st.columns([3.5, 1.7, 2.3])
    with top1:
        equipos = st.columns([0.5, 2.0, 0.7, 0.3, 0.5, 2.0])
        with equipos[0]:
            if row.get("logo_local"):
                st.image(row["logo_local"], width=28)
        with equipos[1]:
            st.markdown(f"<div style='font-size:13px;font-weight:700;color:{COLOR_TEXT_MAIN};'>{local}</div>", unsafe_allow_html=True)
        with equipos[2]:
            marcador_html = f"{marcador_local} - {marcador_visitante}" if mostrar_marcador else "vs"
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_MUTED};text-align:center;font-weight:700;'>{marcador_html}</div>"
                + (
                    f"<div style='font-size:11px;color:{COLOR_TEXT_MUTED};text-align:center;font-weight:700;margin-top:2px;'>{minuto_partido}</div>"
                    if estado_actual in {"LIVE", "HT", "1H", "2H"} and minuto_partido else
                    ""
                ),
                unsafe_allow_html=True,
            )
        with equipos[3]:
            st.markdown("<div></div>", unsafe_allow_html=True)
        with equipos[4]:
            if row.get("logo_visitante"):
                st.image(row["logo_visitante"], width=28)
        with equipos[5]:
            st.markdown(f"<div style='font-size:13px;font-weight:700;color:{COLOR_TEXT_MAIN};'>{visitante}</div>", unsafe_allow_html=True)

        st.markdown(
            f"<div style='font-size:11px;color:{COLOR_TEXT_SOFT};'>"
            f"{row.get('hora_partido', '')} | {badge_estado(row.get('estado_partido', 'NS'))} | "
            f"Pick: <b>{pick_recomendado}</b> · Mercado: <b>{mercado_recomendado}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if consenso:
            score_txt = f" ({consenso_score})" if consenso_score not in ("", None) else ""
            st.markdown(
                f"<div style='font-size:11px;color:{COLOR_TEXT_MUTED};margin-top:4px;'><b>Consenso:</b> {consenso.capitalize()}{score_txt}"
                + (f" · {consenso_notas}" if consenso_notas else "")
                + "</div>",
                unsafe_allow_html=True,
            )

    with top2:
        st.markdown(badge_decision(row.get("decision_simple", "Mirar")), unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:12px;color:{COLOR_TEXT_MAIN};margin-top:8px;'><b>{pick_recomendado}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:11px;color:{COLOR_TEXT_MUTED};margin-top:4px;'><b>{mercado_recomendado}</b></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:11px;color:{COLOR_TEXT_SOFT};margin-top:4px;'><b>{row.get('disciplina_simple', 'Solo seguimiento')}</b> · {row.get('riesgo_simple', 'Riesgo medio')}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:11px;color:{COLOR_TEXT_MUTED};margin-top:4px;'>Prob. estimada: <b>{_fmt_pct(probabilidad_pick)}</b> · Confianza: <b>{confianza_pick or 'Evitar'}</b></div>"
            f"<div style='font-size:11px;color:{COLOR_TEXT_SOFT};margin-top:4px;'>Cuota: <b>{_fmt_num(cuota_pick)}</b> · Edge: <b>{_fmt_pct(edge_pick if not sin_valor_real else None)}</b> · EV: <b>{_fmt_pct(ev_pick if not sin_valor_real else None)}</b></div>",
            unsafe_allow_html=True,
        )

    with top3:
        if top_picks and not sin_valor_real:
            top_lines = "".join(
                f"<div style='font-size:12px;color:{COLOR_TEXT_MAIN};margin-bottom:4px;'><b>{item['label']}:</b> {item['pick']} · {item['mercado']} · <b>{item['prob']}%</b>"
                + (f" · {item['confianza']}" if item["confianza"] else "")
                + (f" · {item['riesgo']}" if item["riesgo"] else "")
                + (f"<div style='font-size:11px;color:{COLOR_TEXT_SOFT};margin-top:2px;'>Cuota {item['cuota']} · Edge {item['edge']}% · EV {item['ev']}% · Score {item['score_final']}</div>" if item["cuota"] not in ("", None) else "")
                + "</div>"
                for item in top_picks
            )
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_MUTED};margin-bottom:6px;'><b>Top mercados del partido</b></div>{top_lines}",
                unsafe_allow_html=True,
            )
        elif sin_valor_real:
            observado = f"{pick_observado} · {mercado_observado}" if pick_observado and mercado_observado else "Sin mercado defendible"
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_MUTED};margin-bottom:6px;'><b>Mejor mercado observado:</b> {observado}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_SOFT};margin-bottom:6px;'><b>Por qué fue descartado:</b> {motivo_descarte or 'Ningún mercado supera los filtros mínimos de probabilidad, cuota, edge, EV y riesgo.'}</div>",
                unsafe_allow_html=True,
            )
        if justificacion_ranking and not sin_valor_real:
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_SOFT};margin-bottom:6px;'><b>Por qué este pick va primero:</b> {justificacion_ranking} · <b>Score final:</b> {score_final}</div>",
                unsafe_allow_html=True,
            )
        if not sin_valor_real:
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_MUTED};'><b>Lectura rápida:</b> {row.get('contexto_simple', 'Sin contexto adicional')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_MAIN};margin-top:6px;'><b>Conclusión:</b> {row.get('razon_simple', '')}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_MAIN};margin-top:6px;'><b>Razón:</b> {motivo_descarte or 'Ningún mercado supera los filtros mínimos de probabilidad, cuota, edge, EV y riesgo.'}</div>",
                unsafe_allow_html=True,
            )
        if claves_pick and not sin_valor_real:
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_SOFT};margin-top:4px;'><b>Claves del pick:</b> {claves_pick}</div>",
                unsafe_allow_html=True,
            )
        if analisis_ia:
            auditor_tags = " · ".join([v for v in [veredicto_ia, decision_ia] if v])
            etiqueta = f" ({auditor_tags})" if auditor_tags else ""
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_MUTED};margin-top:6px;'><b>Auditor IA{etiqueta}:</b> {analisis_ia}</div>",
                unsafe_allow_html=True,
            )
        if mercado_ia:
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_SOFT};margin-top:4px;'><b>Mercado sugerido por auditor:</b> {mercado_ia}</div>",
                unsafe_allow_html=True,
            )
        if claves_ia:
            st.markdown(
                f"<div style='font-size:12px;color:{COLOR_TEXT_SOFT};margin-top:4px;'><b>Hallazgos del auditor:</b> {claves_ia}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


st.set_page_config(layout="wide", page_title="AI Predictor PRO")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 2rem;}
    div[data-testid="stMetricLabel"] {color: var(--text-color) !important;}
    div[data-testid="stCaptionContainer"] {color: color-mix(in srgb, var(--text-color) 68%, transparent) !important;}
    @keyframes livePulse {
        0% { transform: scale(1); opacity: 0.9; box-shadow: 0 0 0 0 rgba(239,68,68,0.25); }
        70% { transform: scale(1.12); opacity: 1; box-shadow: 0 0 0 8px rgba(239,68,68,0); }
        100% { transform: scale(1); opacity: 0.9; box-shadow: 0 0 0 0 rgba(239,68,68,0); }
    }
    div[data-testid="stExpander"] {
        border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
        border-radius: 12px;
        background: color-mix(in srgb, var(--secondary-background-color) 94%, var(--background-color) 6%);
    }
    div[data-testid="stExpander"] details summary p {
        color: var(--text-color) !important;
        font-weight: 700 !important;
    }
    div[data-testid="stTabs"] button p {
        color: color-mix(in srgb, var(--text-color) 78%, transparent) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Predictor PRO")

@st.fragment(run_every=LIVE_REFRESH_INTERVAL)
def render_dashboard():
    top_controls = st.columns([1, 1])
    with top_controls[0]:
        actualizar_historial_btn = st.button("Actualizar historial")
    with top_controls[1]:
        guardar_picks_btn = st.button("Guardar picks")
    perfil_visible = "Agresivo"
    perfil_riesgo = "agresivo"

    if actualizar_historial_btn:
        with st.spinner("Actualizando resultados del historial..."):
            actualizar_resultados()
        cargar_resumen_historial.clear()
        cargar_calibracion_modelo.clear()

    df = cargar_partidos()
    api_status = get_last_api_status()

    if df.empty:
        st.error(api_status.get("message") or "No hay partidos disponibles")
        if api_status.get("details"):
            st.caption(f"Detalle técnico: {api_status['details']}")
        return

    sincronizar_historial_automatico()
    calibracion = cargar_calibracion_modelo()
    resultados = preparar_resultados_ui(cargar_resultados(df, calibracion, perfil_riesgo))
    if openai_analisis_disponible():
        resultados = preparar_resultados_ui(cargar_resultados_openai(resultados))
    stats_historial = cargar_resumen_historial()

    if guardar_picks_btn:
        picks_guardables = resultados[resultados["ganador"] != "No bet"].copy()
        if picks_guardables.empty:
            st.info("Hoy no hay picks suficientemente claros para guardar.")
        else:
            changed = guardar_resultado(picks_guardables)
            cargar_resumen_historial.clear()
            cargar_calibracion_modelo.clear()
            if changed:
                st.success("Picks del día guardados.")
            else:
                st.info("No hubo cambios nuevos para guardar.")

    _, total = calcular_roi()
    en_vivo = int(resultados["estado_partido"].isin(["LIVE", "HT", "1H", "2H"]).sum()) if "estado_partido" in resultados.columns else 0
    proximos = int(len(resultados) - en_vivo)
    picks_claros = int((resultados["decision_simple"] == "Apostar").sum()) if "decision_simple" in resultados.columns else 0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("En vivo", en_vivo)
    with m2:
        st.metric("Próximos", proximos)
    with m3:
        st.metric("Partidos para apostar", picks_claros)

    audit_total = int(resultados["auditoria_inconsistencias"].fillna(0).sum()) if "auditoria_inconsistencias" in resultados.columns else 0
    audit_matches = int(resultados["auditoria_partido_afectado"].fillna(False).astype(bool).sum()) if "auditoria_partido_afectado" in resultados.columns else 0
    audit_markets = 0
    if "auditoria_mercados_corregidos" in resultados.columns:
        corrected = set()
        for raw in resultados["auditoria_mercados_corregidos"].fillna(""):
            for item in [x.strip() for x in str(raw).split(",") if x.strip()]:
                corrected.add(item)
        audit_markets = len(corrected)

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Inconsistencias corregidas", audit_total)
    with a2:
        st.metric("Partidos afectados", audit_matches)
    with a3:
        st.metric("Mercados corregidos", audit_markets)

    if openai_analisis_disponible():
        st.caption("Socio analista IA activo como apoyo extra.")

    tab1, tab2 = st.tabs(["Partidos por liga", "Historial"])

    with tab1:
        destacados = seleccionar_partidos_destacados(resultados, max_items=None)
        if not destacados.empty:
            st.markdown("### Para mirar primero")
            render_grid_tarjetas(destacados, mostrar_disciplina=True, cards_per_row=3)

        en_vivo_cards = seleccionar_partidos_en_vivo(resultados, max_items=6)
        if not en_vivo_cards.empty:
            st.markdown("### En vivo ahora")
            render_grid_tarjetas(en_vivo_cards, mostrar_disciplina=False, cards_per_row=3)

        ligas_ordenadas = (
            resultados.groupby(["liga", "grupo_liga"], dropna=False)
            .agg(partidos=("partido", "size"), prioridad_liga=("prioridad_liga", "max"))
            .reset_index()
            .sort_values(by=["prioridad_liga", "grupo_liga", "liga"], ascending=[False, True, True])
        )

        for index_liga, (_, liga_row) in enumerate(ligas_ordenadas.iterrows()):
            liga = liga_row["liga"]
            grupo = liga_row["grupo_liga"]
            subset = resultados[resultados["liga"] == liga].copy()
            subset = subset.sort_values(
                by=["decision_rank", "hora_partido", "partido"],
                ascending=[True, True, True],
            )

            titulo = f"{grupo or liga} ({len(subset)} partidos)"
            with st.expander(titulo, expanded=index_liga < 2):
                for _, row in subset.iterrows():
                    render_partido(row)

    with tab2:
        st.subheader("Historial simple")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Cerradas", stats_historial["cerradas"])
        with c2:
            st.metric("Pendientes", stats_historial["pendientes"])
        with c3:
            st.metric("Aciertos", stats_historial["aciertos"])
        with c4:
            st.metric("Total apostado", f"{total}u")

        valid1, valid2, valid3, valid4 = st.columns(4)
        metricas_apostar = stats_historial.get("metricas_apostar", {})
        metricas_mirar = stats_historial.get("metricas_mirar", {})
        with valid1:
            st.metric("Apostar cerradas", metricas_apostar.get("picks", 0))
        with valid2:
            st.metric("Acierto en Apostar", f"{metricas_apostar.get('hit_rate', 0)}%")
        with valid3:
            st.metric("Mirar cerradas", metricas_mirar.get("picks", 0))
        with valid4:
            st.metric("Acierto en Mirar", f"{metricas_mirar.get('hit_rate', 0)}%")

        learn1, learn2, learn3 = st.columns(3)
        with learn1:
            st.metric("Precisión predicción", f"{stats_historial.get('precision_prediccion_simple', 0)}%")
        with learn2:
            st.metric("Profit Apostar", f"{metricas_apostar.get('profit', 0)}u")
        with learn3:
            st.metric("Profit Mirar", f"{metricas_mirar.get('profit', 0)}u")

        st.markdown("### Validación rápida")
        segmentos_decision = stats_historial.get("segmentos_decision")
        if segmentos_decision is not None and not segmentos_decision.empty:
            columnas_decision = [
                col for col in [
                    "decision_simple",
                    "picks",
                    "aciertos",
                    "fallos",
                    "hit_rate",
                    "roi",
                    "profit",
                    "nivel_riesgo",
                ]
                if col in segmentos_decision.columns
            ]
            st.dataframe(
                segmentos_decision[columnas_decision],
                use_container_width=True,
                hide_index=True,
            )

        segmentos_prediccion = stats_historial.get("segmentos_prediccion")
        if segmentos_prediccion is not None and not segmentos_prediccion.empty:
            st.markdown("### Rendimiento por predicción")
            columnas_pred = [
                col for col in [
                    "prediccion_simple",
                    "picks",
                    "aciertos",
                    "fallos",
                    "hit_rate",
                    "roi",
                    "profit",
                    "nivel_riesgo",
                ]
                if col in segmentos_prediccion.columns
            ]
            st.dataframe(
                segmentos_prediccion[columnas_pred],
                use_container_width=True,
                hide_index=True,
            )

        segmentos_liga = stats_historial.get("segmentos_liga")
        if segmentos_liga is not None and not segmentos_liga.empty:
            st.markdown("### Rendimiento por liga")
            columnas_liga = [
                col for col in [
                    "liga",
                    "picks",
                    "aciertos",
                    "fallos",
                    "hit_rate",
                    "roi",
                    "profit",
                    "nivel_riesgo",
                ]
                if col in segmentos_liga.columns
            ]
            st.dataframe(
                segmentos_liga[columnas_liga],
                use_container_width=True,
                hide_index=True,
            )

        contexto_col1, contexto_col2 = st.columns(2)
        segmentos_contexto = stats_historial.get("segmentos_contexto")
        if segmentos_contexto is not None and not segmentos_contexto.empty:
            with contexto_col1:
                st.markdown("### Rendimiento por contexto")
                columnas_contexto = [
                    col for col in [
                        "contexto_partido",
                        "picks",
                        "aciertos",
                        "fallos",
                        "hit_rate",
                        "roi",
                        "profit",
                    ]
                    if col in segmentos_contexto.columns
                ]
                st.dataframe(
                    segmentos_contexto[columnas_contexto],
                    use_container_width=True,
                    hide_index=True,
                )

        segmentos_estado_consulta = stats_historial.get("segmentos_estado_consulta")
        if segmentos_estado_consulta is not None and not segmentos_estado_consulta.empty:
            with contexto_col2:
                st.markdown("### Estado de cierres")
                columnas_cierre = [
                    col for col in [
                        "estado_final_consulta",
                        "picks",
                    ]
                    if col in segmentos_estado_consulta.columns
                ]
                st.dataframe(
                    segmentos_estado_consulta[columnas_cierre],
                    use_container_width=True,
                    hide_index=True,
                )

        top_col, bottom_col = st.columns(2)
        top_ligas = stats_historial.get("top_ligas")
        if top_ligas is not None and not top_ligas.empty:
            with top_col:
                st.markdown("### Ligas más fuertes")
                st.dataframe(
                    top_ligas[[col for col in ["liga", "picks", "hit_rate", "roi", "profit"] if col in top_ligas.columns]],
                    use_container_width=True,
                    hide_index=True,
                )

        bottom_ligas = stats_historial.get("bottom_ligas")
        if bottom_ligas is not None and not bottom_ligas.empty:
            with bottom_col:
                st.markdown("### Ligas más flojas")
                st.dataframe(
                    bottom_ligas[[col for col in ["liga", "picks", "hit_rate", "roi", "profit"] if col in bottom_ligas.columns]],
                    use_container_width=True,
                    hide_index=True,
                )

        historial_df = stats_historial["historial"]
        if historial_df.empty:
            st.info("Aún no hay historial guardado.")
        else:
            columnas = [
                col for col in [
                    "fecha_partido",
                    "liga",
                    "partido",
                    "prediccion_simple",
                    "decision_simple",
                    "resultado_real",
                    "estado_final_consulta",
                    "profit",
                ]
                if col in historial_df.columns
            ]
            vista = historial_df.copy()
            if "fecha_partido" in vista.columns:
                vista = vista.sort_values(by="fecha_partido", ascending=False, na_position="last")
                vista["fecha_partido"] = vista["fecha_partido"].dt.strftime("%Y-%m-%d")
            st.dataframe(vista[columnas], use_container_width=True, hide_index=True)


render_dashboard()
