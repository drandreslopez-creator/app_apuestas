import pandas as pd
import numpy as np
import json
import os
import sqlite3
from datetime import datetime

import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVO = os.path.join(BASE_DIR, "historial.csv")
DB_PATH = os.path.join(BASE_DIR, "app_apuestas.db")
SUPABASE_TABLE = "ai_predictor_history"
_CLOUD_HISTORY_LOADED = False
_CLOUD_STATUS = "local"

COLUMNAS_HISTORIAL = [
    "fixture_id",
    "fecha_partido",
    "hora_partido",
    "liga",
    "grupo_liga",
    "partido",
    "ganador",
    "mercado",
    "confianza",
    "estado_partido",
    "fuente_cuotas",
    "value_local",
    "value_empate",
    "value_visitante",
    "prob_local",
    "prob_empate",
    "prob_visitante",
    "ai_score",
    "ai_decision",
    "ai_resumen",
    "mejor_edge_pct",
    "ev_por_unidad",
    "banda_confianza",
    "banda_edge",
    "cuota_local",
    "cuota_empate",
    "cuota_visitante",
    "stake",
    "stake_num",
    "fecha_guardado",
    "resultado_real",
    "fecha_cierre_resultado",
    "estado_final_consulta",
    "fuente_partidos",
    "prediccion_simple",
    "decision_simple",
    "razon_simple",
    "pick_recomendado",
    "mercado_recomendado",
    "probabilidad_pick",
    "riesgo_pick",
    "odds_source_pick",
]

ESTADOS_EN_VIVO = {"1H", "HT", "2H", "ET", "BT", "P", "SUSP", "INT", "LIVE"}
ESTADOS_NO_INICIADOS = {"NS", "TBD", "PST"}
ESTADOS_CERRADOS = {"FT", "AET", "PEN", "CANC", "ABD", "AWD", "WO"}


def _get_setting(name):
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        return None


def _supabase_config():
    url = str(_get_setting("SUPABASE_URL") or "").strip().rstrip("/")
    key = str(_get_setting("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url or not key:
        return None
    return {"url": url, "key": key}


def estado_persistencia():
    """Expose a safe status label without ever exposing cloud credentials."""
    if _supabase_config() is None:
        return "Historial local"
    return "Historial persistente" if _CLOUD_STATUS == "persistent" else "Historial local (nube pendiente)"


def _json_safe(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _cargar_historial_nube(conn):
    global _CLOUD_HISTORY_LOADED, _CLOUD_STATUS
    if _CLOUD_HISTORY_LOADED:
        return
    _CLOUD_HISTORY_LOADED = True
    config = _supabase_config()
    if config is None:
        return

    try:
        response = requests.get(
            f"{config['url']}/rest/v1/{SUPABASE_TABLE}",
            params={"select": "unique_key,payload"},
            headers={"apikey": config["key"], "Authorization": f"Bearer {config['key']}"},
            timeout=8,
        )
        response.raise_for_status()
        rows = response.json()
        payloads = [item.get("payload") for item in rows if isinstance(item.get("payload"), dict)]
        if payloads:
            remoto = _asegurar_columnas(pd.DataFrame(payloads))
            remoto["unique_key"] = remoto.apply(_build_unique_key, axis=1)
            remoto = remoto.drop_duplicates(subset=["unique_key"], keep="last")
            conn.executemany(
                "INSERT OR IGNORE INTO historial (unique_key, fixture_id, fecha_partido, hora_partido, liga, grupo_liga, partido, ganador, mercado, confianza, estado_partido, fuente_cuotas, value_local, value_empate, value_visitante, prob_local, prob_empate, prob_visitante, ai_score, ai_decision, ai_resumen, mejor_edge_pct, ev_por_unidad, banda_confianza, banda_edge, cuota_local, cuota_empate, cuota_visitante, stake, stake_num, fecha_guardado, resultado_real, fecha_cierre_resultado, estado_final_consulta, fuente_partidos, prediccion_simple, decision_simple, razon_simple, pick_recomendado, mercado_recomendado, probabilidad_pick, riesgo_pick, odds_source_pick) VALUES (:unique_key, :fixture_id, :fecha_partido, :hora_partido, :liga, :grupo_liga, :partido, :ganador, :mercado, :confianza, :estado_partido, :fuente_cuotas, :value_local, :value_empate, :value_visitante, :prob_local, :prob_empate, :prob_visitante, :ai_score, :ai_decision, :ai_resumen, :mejor_edge_pct, :ev_por_unidad, :banda_confianza, :banda_edge, :cuota_local, :cuota_empate, :cuota_visitante, :stake, :stake_num, :fecha_guardado, :resultado_real, :fecha_cierre_resultado, :estado_final_consulta, :fuente_partidos, :prediccion_simple, :decision_simple, :razon_simple, :pick_recomendado, :mercado_recomendado, :probabilidad_pick, :riesgo_pick, :odds_source_pick)",
                remoto.to_dict(orient="records"),
            )
            conn.commit()
        _CLOUD_STATUS = "persistent"
    except (requests.RequestException, ValueError, sqlite3.Error):
        _CLOUD_STATUS = "unavailable"


def _guardar_historial_nube(df):
    global _CLOUD_STATUS
    config = _supabase_config()
    if config is None or df.empty:
        return False

    registros = []
    for row in df.to_dict(orient="records"):
        unique_key = str(row.get("unique_key") or _build_unique_key(row))
        payload = {key: _json_safe(value) for key, value in row.items() if key != "unique_key"}
        registros.append({"unique_key": unique_key, "payload": payload})
    try:
        response = requests.post(
            f"{config['url']}/rest/v1/{SUPABASE_TABLE}",
            params={"on_conflict": "unique_key"},
            headers={
                "apikey": config["key"],
                "Authorization": f"Bearer {config['key']}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
            data=json.dumps(registros, ensure_ascii=False),
            timeout=8,
        )
        response.raise_for_status()
        _CLOUD_STATUS = "persistent"
        return True
    except (requests.RequestException, TypeError, ValueError):
        _CLOUD_STATUS = "unavailable"
        return False


def _prediccion_simple_desde_ganador(ganador):
    mapa = {
        "Gana local": "Gana el local",
        "Gana visitante": "Gana el visitante",
        "Empate": "Empate",
        "No bet": "Partido para mirar",
    }
    valor = str(ganador or "").strip()
    return mapa.get(valor, "Partido para mirar")


def _get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _asegurar_columnas(df):
    df_ajustado = df.copy()
    if "fecha_guardado" not in df_ajustado.columns:
        df_ajustado["fecha_guardado"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "estado_final_consulta" not in df_ajustado.columns:
        df_ajustado["estado_final_consulta"] = "pendiente"

    for col in COLUMNAS_HISTORIAL:
        if col not in df_ajustado.columns:
            df_ajustado[col] = None

    if "prediccion_simple" in df_ajustado.columns and "ganador" in df_ajustado.columns:
        pred_series = df_ajustado["prediccion_simple"].fillna("").astype(str).str.strip()
        missing_pred = pred_series.eq("")
        if missing_pred.any():
            df_ajustado.loc[missing_pred, "prediccion_simple"] = df_ajustado.loc[missing_pred, "ganador"].apply(
                _prediccion_simple_desde_ganador
            )

    if "resultado_real" in df_ajustado.columns and "estado_final_consulta" in df_ajustado.columns:
        mask_cerrado = df_ajustado["resultado_real"].notna() & df_ajustado["resultado_real"].astype(str).str.strip().ne("")
        df_ajustado.loc[mask_cerrado & df_ajustado["estado_final_consulta"].fillna("").astype(str).str.strip().eq(""), "estado_final_consulta"] = "cerrado_con_resultado"
        df_ajustado.loc[mask_cerrado & df_ajustado["estado_final_consulta"].fillna("").astype(str).str.strip().eq("pendiente"), "estado_final_consulta"] = "cerrado_con_resultado"

    for col in ["fecha_partido", "hora_partido", "fecha_guardado", "fecha_cierre_resultado"]:
        if col in df_ajustado.columns:
            df_ajustado[col] = df_ajustado[col].apply(
                lambda x: None if pd.isna(x) else (
                    x.strftime("%Y-%m-%d") if col == "fecha_partido" and hasattr(x, "strftime")
                    else x.strftime("%Y-%m-%d %H:%M") if col == "hora_partido" and hasattr(x, "strftime")
                    else x.strftime("%Y-%m-%d %H:%M:%S") if col in {"fecha_guardado", "fecha_cierre_resultado"} and hasattr(x, "strftime")
                    else str(x)
                )
            )

    df_ajustado = df_ajustado[COLUMNAS_HISTORIAL]
    return df_ajustado


def _build_unique_key(row):
    fixture_id = row.get("fixture_id")
    if pd.notna(fixture_id) and str(fixture_id).strip():
        return f"fixture:{str(fixture_id).strip()}"
    fecha = str(row.get("fecha_partido") or "").strip()
    partido = str(row.get("partido") or "").strip()
    return f"match:{fecha}:{partido}"


def _clasificar_contexto_partido(estado_partido):
    estado = str(estado_partido or "").strip().upper()
    if estado in ESTADOS_EN_VIVO:
        return "En vivo"
    if estado in ESTADOS_NO_INICIADOS:
        return "Prepartido"
    if estado in ESTADOS_CERRADOS:
        return "Finalizado"
    return "Sin dato"


def init_storage():
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS historial (
                unique_key TEXT PRIMARY KEY,
                fixture_id TEXT,
                fecha_partido TEXT,
                hora_partido TEXT,
                liga TEXT,
                grupo_liga TEXT,
                partido TEXT,
                ganador TEXT,
                mercado TEXT,
                confianza REAL,
                estado_partido TEXT,
                fuente_cuotas TEXT,
                value_local TEXT,
                value_empate TEXT,
                value_visitante TEXT,
                prob_local REAL,
                prob_empate REAL,
                prob_visitante REAL,
                ai_score REAL,
                ai_decision TEXT,
                ai_resumen TEXT,
                mejor_edge_pct REAL,
                ev_por_unidad REAL,
                banda_confianza TEXT,
                banda_edge TEXT,
                cuota_local REAL,
                cuota_empate REAL,
                cuota_visitante REAL,
                stake TEXT,
                stake_num REAL,
                fecha_guardado TEXT,
                resultado_real TEXT,
                fecha_cierre_resultado TEXT,
                estado_final_consulta TEXT,
                fuente_partidos TEXT,
                prediccion_simple TEXT,
                decision_simple TEXT,
                razon_simple TEXT,
                pick_recomendado TEXT,
                mercado_recomendado TEXT,
                probabilidad_pick REAL,
                riesgo_pick TEXT,
                odds_source_pick TEXT
            )
            """
        )

        columnas_actuales = {
            row[1] for row in conn.execute("PRAGMA table_info(historial)").fetchall()
        }
        columnas_requeridas = {
            "ai_score": "REAL",
            "ai_decision": "TEXT",
            "ai_resumen": "TEXT",
            "mejor_edge_pct": "REAL",
            "ev_por_unidad": "REAL",
            "banda_confianza": "TEXT",
            "banda_edge": "TEXT",
            "fuente_partidos": "TEXT",
            "grupo_liga": "TEXT",
            "prediccion_simple": "TEXT",
            "decision_simple": "TEXT",
            "razon_simple": "TEXT",
            "fecha_cierre_resultado": "TEXT",
            "estado_final_consulta": "TEXT",
            "pick_recomendado": "TEXT",
            "mercado_recomendado": "TEXT",
            "probabilidad_pick": "REAL",
            "riesgo_pick": "TEXT",
            "odds_source_pick": "TEXT",
        }
        for columna, tipo in columnas_requeridas.items():
            if columna not in columnas_actuales:
                conn.execute(f"ALTER TABLE historial ADD COLUMN {columna} {tipo}")

        _cargar_historial_nube(conn)

        total = conn.execute("SELECT COUNT(*) FROM historial").fetchone()[0]
        if total == 0 and os.path.exists(ARCHIVO):
            try:
                df_csv = pd.read_csv(ARCHIVO)
                if not df_csv.empty:
                    df_csv = _asegurar_columnas(df_csv)
                    _upsert_historial(df_csv, conn=conn)
            except Exception:
                pass


def _upsert_historial(df, conn=None, preservar_snapshot=False):
    df_guardar = _asegurar_columnas(df)
    if df_guardar.empty:
        return False

    df_guardar = df_guardar.copy()
    df_guardar["unique_key"] = df_guardar.apply(_build_unique_key, axis=1)
    df_guardar = df_guardar.drop_duplicates(subset=["unique_key"], keep="last")

    close_conn = False
    if conn is None:
        conn = _get_connection()
        close_conn = True

    if preservar_snapshot:
        existentes = {
            row[0] for row in conn.execute("SELECT unique_key FROM historial").fetchall()
        }
        df_guardar = df_guardar[~df_guardar["unique_key"].isin(existentes)].copy()
        if df_guardar.empty:
            if close_conn:
                conn.close()
            return False

    before = conn.execute("SELECT COUNT(*) FROM historial").fetchone()[0]
    conn.executemany(
        """
        INSERT INTO historial (
            unique_key, fixture_id, fecha_partido, hora_partido, liga, grupo_liga, partido,
            ganador, mercado, confianza, estado_partido, fuente_cuotas,
            value_local, value_empate, value_visitante,
            prob_local, prob_empate, prob_visitante, ai_score, ai_decision, ai_resumen, mejor_edge_pct, ev_por_unidad, banda_confianza, banda_edge,
            cuota_local, cuota_empate, cuota_visitante,
            stake, stake_num, fecha_guardado, resultado_real, fecha_cierre_resultado, estado_final_consulta, fuente_partidos
            , prediccion_simple, decision_simple, razon_simple,
            pick_recomendado, mercado_recomendado, probabilidad_pick, riesgo_pick, odds_source_pick
        ) VALUES (
            :unique_key, :fixture_id, :fecha_partido, :hora_partido, :liga, :grupo_liga, :partido,
            :ganador, :mercado, :confianza, :estado_partido, :fuente_cuotas,
            :value_local, :value_empate, :value_visitante,
            :prob_local, :prob_empate, :prob_visitante, :ai_score, :ai_decision, :ai_resumen, :mejor_edge_pct, :ev_por_unidad, :banda_confianza, :banda_edge,
            :cuota_local, :cuota_empate, :cuota_visitante,
            :stake, :stake_num, :fecha_guardado, :resultado_real, :fecha_cierre_resultado, :estado_final_consulta, :fuente_partidos,
            :prediccion_simple, :decision_simple, :razon_simple,
            :pick_recomendado, :mercado_recomendado, :probabilidad_pick, :riesgo_pick, :odds_source_pick
        )
        ON CONFLICT(unique_key) DO UPDATE SET
            fixture_id=excluded.fixture_id,
            fecha_partido=excluded.fecha_partido,
            hora_partido=excluded.hora_partido,
            liga=excluded.liga,
            grupo_liga=excluded.grupo_liga,
            partido=excluded.partido,
            ganador=excluded.ganador,
            mercado=excluded.mercado,
            confianza=excluded.confianza,
            estado_partido=excluded.estado_partido,
            fuente_cuotas=excluded.fuente_cuotas,
            value_local=excluded.value_local,
            value_empate=excluded.value_empate,
            value_visitante=excluded.value_visitante,
            prob_local=excluded.prob_local,
            prob_empate=excluded.prob_empate,
            prob_visitante=excluded.prob_visitante,
            ai_score=excluded.ai_score,
            ai_decision=excluded.ai_decision,
            ai_resumen=excluded.ai_resumen,
            mejor_edge_pct=excluded.mejor_edge_pct,
            ev_por_unidad=excluded.ev_por_unidad,
            banda_confianza=excluded.banda_confianza,
            banda_edge=excluded.banda_edge,
            cuota_local=excluded.cuota_local,
            cuota_empate=excluded.cuota_empate,
            cuota_visitante=excluded.cuota_visitante,
            stake=excluded.stake,
            stake_num=excluded.stake_num,
            fecha_guardado=excluded.fecha_guardado,
            resultado_real=COALESCE(historial.resultado_real, excluded.resultado_real),
            fecha_cierre_resultado=COALESCE(historial.fecha_cierre_resultado, excluded.fecha_cierre_resultado),
            estado_final_consulta=excluded.estado_final_consulta,
            fuente_partidos=excluded.fuente_partidos,
            prediccion_simple=excluded.prediccion_simple,
            decision_simple=excluded.decision_simple,
            razon_simple=excluded.razon_simple,
            pick_recomendado=excluded.pick_recomendado,
            mercado_recomendado=excluded.mercado_recomendado,
            probabilidad_pick=excluded.probabilidad_pick,
            riesgo_pick=excluded.riesgo_pick,
            odds_source_pick=excluded.odds_source_pick
        """,
        df_guardar.to_dict(orient="records")
    )
    after = conn.execute("SELECT COUNT(*) FROM historial").fetchone()[0]
    changed = conn.total_changes > 0 or before != after
    conn.commit()
    _guardar_historial_nube(df_guardar)

    if close_conn:
        conn.close()

    return changed


def guardar_resultado(df):
    init_storage()
    # The first pre-match read is immutable: later refreshes must not rewrite
    # the evidence used to evaluate the prediction.
    return _upsert_historial(df, preservar_snapshot=True)


# ================================
# 🔥 ACTUALIZAR RESULTADOS AUTOMÁTICOS
# ================================

def actualizar_resultados():
    init_storage()

    df = cargar_historial()

    pendientes = df[df["resultado_real"].isna()]

    if pendientes.empty:
        return False

    from api_data import buscar_resultado_partido

    actualizado = False
    cache_consultas = {}

    for i, row in df.iterrows():

        if pd.notna(row["resultado_real"]):
            continue

        fecha = row.get("fecha_partido")
        if pd.isna(fecha) or str(fecha).strip() == "":
            fecha = pd.Timestamp.now().strftime("%Y-%m-%d")
        else:
            fecha = pd.to_datetime(fecha, errors="coerce")
            fecha = fecha.strftime("%Y-%m-%d") if pd.notna(fecha) else pd.Timestamp.now().strftime("%Y-%m-%d")

        fixture_id = row.get("fixture_id")
        partido = row.get("partido")
        cache_key = (str(fixture_id or ""), str(fecha), str(partido or ""))

        if cache_key not in cache_consultas:
            cache_consultas[cache_key] = buscar_resultado_partido(
                fecha=fecha,
                fixture_id=str(fixture_id).strip() if pd.notna(fixture_id) and str(fixture_id).strip() else None,
                partido=partido,
            )

        match = cache_consultas.get(cache_key)
        if not match:
            if str(row.get("estado_final_consulta") or "") != "sin_respuesta_proveedor":
                df.at[i, "estado_final_consulta"] = "sin_respuesta_proveedor"
                actualizado = True
            continue

        resultado = match.get("resultado_real")
        cambios_fila = False

        if pd.isna(row.get("fixture_id")) or str(row.get("fixture_id")).strip() == "":
            nuevo_fixture = match.get("fixture_id")
            if pd.notna(nuevo_fixture) and str(nuevo_fixture).strip():
                df.at[i, "fixture_id"] = nuevo_fixture
                cambios_fila = True

        if pd.isna(row.get("fecha_partido")) or str(row.get("fecha_partido")).strip() == "":
            nueva_fecha = match.get("fecha_partido")
            if nueva_fecha:
                df.at[i, "fecha_partido"] = nueva_fecha
                cambios_fila = True

        if pd.isna(row.get("hora_partido")) or str(row.get("hora_partido")).strip() == "":
            nueva_hora = match.get("hora_partido")
            if nueva_hora:
                df.at[i, "hora_partido"] = nueva_hora
                cambios_fila = True

        if pd.isna(row.get("liga")) or str(row.get("liga")).strip() == "":
            nueva_liga = match.get("liga")
            if nueva_liga:
                df.at[i, "liga"] = nueva_liga
                cambios_fila = True

        nuevo_estado = match.get("estado_partido")
        if nuevo_estado and str(row.get("estado_partido") or "") != str(nuevo_estado):
            df.at[i, "estado_partido"] = nuevo_estado
            cambios_fila = True

        if pd.notna(resultado) and str(resultado).strip():
            if pd.isna(row.get("resultado_real")) or str(row.get("resultado_real")).strip() != str(resultado):
                df.at[i, "resultado_real"] = resultado
                if pd.isna(row.get("fecha_cierre_resultado")) or str(row.get("fecha_cierre_resultado")).strip() == "":
                    df.at[i, "fecha_cierre_resultado"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.at[i, "estado_final_consulta"] = "cerrado_con_resultado"
                cambios_fila = True
        else:
            estado_norm = str(nuevo_estado or "").upper()
            if estado_norm in ESTADOS_EN_VIVO:
                nuevo_estado_consulta = "partido_en_juego"
            elif estado_norm in ESTADOS_NO_INICIADOS:
                nuevo_estado_consulta = "pendiente"
            elif estado_norm in ESTADOS_CERRADOS:
                nuevo_estado_consulta = "cerrado_sin_resultado"
            else:
                nuevo_estado_consulta = "sin_resultado_final"

            if str(row.get("estado_final_consulta") or "") != nuevo_estado_consulta:
                df.at[i, "estado_final_consulta"] = nuevo_estado_consulta
                cambios_fila = True

        if cambios_fila:
            actualizado = True

    if not actualizado:
        return False

    df_actualizar = _asegurar_columnas(df)
    return _upsert_historial(df_actualizar, preservar_snapshot=False)


# ================================
# 🔥 ROI REAL
# ================================

def calcular_roi():
    df = cargar_historial()

    df = df.dropna(subset=["resultado_real"])
    if "es_apuesta_real" in df.columns:
        df = df[df["es_apuesta_real"]]

    if df.empty:
        return 0, 0

    profit_total = 0
    stake_total = 0

    for _, row in df.iterrows():

        # 🔥 FIX STAKE (BIEN INDENTADO)
        stake_raw = row.get("stake_num", 1)

        try:
            stake = float(stake_raw)
            if pd.isna(stake):
                stake = 1
        except:
            stake = 1

        if stake <= 0:
            continue

        # 🔥 RESULTADO
        if row["ganador"] == row["resultado_real"]:

            if row["ganador"] == "Gana local":
                cuota = row["cuota_local"]

            elif row["ganador"] == "Gana visitante":
                cuota = row["cuota_visitante"]

            else:
                cuota = row["cuota_empate"]

            profit = (cuota - 1) * stake

        else:
            profit = -stake

        profit_total += profit
        stake_total += stake

    roi = (profit_total / stake_total) * 100 if stake_total > 0 else 0

    return round(roi, 2), round(stake_total, 2)


def calcular_profit_fila(row):

    if not _es_apuesta_real(row):
        return 0.0

    stake_raw = row.get("stake_num", 1)

    try:
        stake = float(stake_raw)
        if pd.isna(stake):
            stake = 1
    except Exception:
        stake = 1

    if stake <= 0:
        return 0.0

    if pd.isna(row.get("resultado_real")):
        return 0.0

    if row["ganador"] == row["resultado_real"]:
        if row["ganador"] == "Gana local":
            cuota = row["cuota_local"]
        elif row["ganador"] == "Gana visitante":
            cuota = row["cuota_visitante"]
        else:
            cuota = row["cuota_empate"]

        return round((float(cuota) - 1) * stake, 2)

    return round(-stake, 2)


def _es_apuesta_real(row):
    return (
        str(row.get("decision_simple") or "").strip() == "Apostar"
        and str(row.get("odds_source_pick") or "").strip().lower() == "real"
    )


def _resultado_prediccion(row):
    if pd.isna(row.get("resultado_real")):
        return "Pendiente"
    pred_map = {
        "Gana el local": "Gana local",
        "Gana el visitante": "Gana visitante",
        "Empate": "Empate",
    }
    prediccion = pred_map.get(str(row.get("prediccion_simple") or "").strip())
    if not prediccion:
        return "Sin lectura"
    return "Acierto" if prediccion == row.get("resultado_real") else "Fallo"


def _metricas_probabilisticas(cerradas):
    """Evaluate the saved 1X2 forecast without treating it as betting ROI."""
    if cerradas.empty:
        return {"muestra": 0, "brier_score": None, "log_loss": None, "calibracion": pd.DataFrame()}

    filas = []
    for _, row in cerradas.iterrows():
        try:
            probs = np.array([
                float(row.get("prob_local") or 0) / 100,
                float(row.get("prob_empate") or 0) / 100,
                float(row.get("prob_visitante") or 0) / 100,
            ])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            continue
        probs = np.clip(probs / probs.sum(), 1e-8, 1.0)
        resultado = str(row.get("resultado_real") or "")
        indice = {"Gana local": 0, "Empate": 1, "Gana visitante": 2}.get(resultado)
        if indice is None:
            continue
        real = np.zeros(3)
        real[indice] = 1.0
        top = float(probs.max())
        filas.append({
            "brier": float(np.mean((probs - real) ** 2)),
            "log_loss": float(-np.log(probs[indice])),
            "prob_top": top,
            "acierto_top": int(int(probs.argmax()) == indice),
        })
    if not filas:
        return {"muestra": 0, "brier_score": None, "log_loss": None, "calibracion": pd.DataFrame()}

    metricas = pd.DataFrame(filas)
    metricas["banda"] = pd.cut(
        metricas["prob_top"],
        bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        labels=["0-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-100%"],
        include_lowest=True,
    )
    calibracion = metricas.groupby("banda", observed=False).agg(
        partidos=("acierto_top", "size"),
        prob_media=("prob_top", "mean"),
        acierto_real=("acierto_top", "mean"),
    ).reset_index()
    calibracion = calibracion[calibracion["partidos"] > 0].copy()
    calibracion["prob_media"] = (calibracion["prob_media"] * 100).round(1)
    calibracion["acierto_real"] = (calibracion["acierto_real"] * 100).round(1)
    return {
        "muestra": int(len(metricas)),
        "brier_score": round(float(metricas["brier"].mean()), 4),
        "log_loss": round(float(metricas["log_loss"].mean()), 4),
        "calibracion": calibracion,
    }


def _max_drawdown(cerradas):
    if cerradas.empty or "profit" not in cerradas.columns:
        return 0.0
    apuestas = cerradas[cerradas.get("es_apuesta_real", False)].copy()
    if apuestas.empty:
        return 0.0
    apuestas = apuestas.sort_values(by="fecha_partido", na_position="last")
    curva = pd.to_numeric(apuestas["profit"], errors="coerce").fillna(0).cumsum()
    pico = curva.cummax()
    return round(float((curva - pico).min()), 2)


def cargar_historial():
    init_storage()

    with _get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM historial", conn)

    if df.empty:
        return df

    if "prediccion_simple" in df.columns and "ganador" in df.columns:
        pred_series = df["prediccion_simple"].fillna("").astype(str).str.strip()
        missing_pred = pred_series.eq("")
        if missing_pred.any():
            df.loc[missing_pred, "prediccion_simple"] = df.loc[missing_pred, "ganador"].apply(
                _prediccion_simple_desde_ganador
            )

    if "fecha_partido" in df.columns:
        df["fecha_partido"] = pd.to_datetime(df["fecha_partido"], errors="coerce")

    if "resultado_real" in df.columns:
        df["es_apuesta_real"] = df.apply(_es_apuesta_real, axis=1)
        df["estado_prediccion"] = df.apply(_resultado_prediccion, axis=1)
        df["estado_pick"] = df.apply(
            lambda row: row["estado_prediccion"] if row["es_apuesta_real"] else "Seguimiento",
            axis=1,
        )
        stake_num = pd.to_numeric(df.get("stake_num", 0), errors="coerce").fillna(0)
        df["stake_contable"] = stake_num.where(df["es_apuesta_real"], 0.0)
        df["profit"] = df.apply(calcular_profit_fila, axis=1)

    if "estado_partido" in df.columns:
        df["contexto_partido"] = df["estado_partido"].apply(_clasificar_contexto_partido)
    else:
        df["contexto_partido"] = "Sin dato"

    if "confianza" in df.columns:
        confianza_num = pd.to_numeric(df["confianza"], errors="coerce").fillna(0)
        df["banda_confianza"] = confianza_num.apply(
            lambda x: "elite" if x >= 80 else "alta" if x >= 70 else "media" if x >= 60 else "baja"
        )

    if "mejor_edge_pct" in df.columns:
        edge_num = pd.to_numeric(df["mejor_edge_pct"], errors="coerce").fillna(0)
        df["banda_edge"] = edge_num.apply(
            lambda x: "elite" if x >= 10 else "alta" if x >= 6 else "media" if x >= 3 else "baja"
        )

    if "ai_score" in df.columns:
        ai_num = pd.to_numeric(df["ai_score"], errors="coerce").fillna(0)
        df["banda_ai"] = ai_num.apply(
            lambda x: "elite" if x >= 82 else "fuerte" if x >= 70 else "observar" if x >= 58 else "descartar"
        )

    return df


def _resumen_segmento(df, columna, min_muestra=3):
    if df.empty or columna not in df.columns:
        return pd.DataFrame()

    segmento_df = df.copy()
    segmento_df[columna] = segmento_df[columna].fillna("Sin dato").astype(str).str.strip()
    segmento_df = segmento_df[segmento_df[columna] != ""]

    if segmento_df.empty:
        return pd.DataFrame()

    estado_columna = "estado_prediccion" if "estado_prediccion" in segmento_df.columns else "estado_pick"
    stake_columna = "stake_contable" if "stake_contable" in segmento_df.columns else "stake_num"
    resumen = (
        segmento_df.groupby(columna, dropna=False)
        .agg(
            picks=("partido", "count"),
            aciertos=(estado_columna, lambda s: int((s == "Acierto").sum())),
            fallos=(estado_columna, lambda s: int((s == "Fallo").sum())),
            profit=("profit", "sum"),
            stake_total=(stake_columna, "sum"),
        )
        .reset_index()
    )

    resumen["hit_rate"] = resumen.apply(
        lambda row: round((row["aciertos"] / row["picks"]) * 100, 2) if row["picks"] else 0.0,
        axis=1,
    )
    resumen["roi"] = resumen.apply(
        lambda row: round((row["profit"] / row["stake_total"]) * 100, 2)
        if row["stake_total"] else 0.0,
        axis=1,
    )
    resumen["nivel_riesgo"] = resumen.apply(
        lambda row: (
            "alto" if row["picks"] >= min_muestra and (row["roi"] < -15 or row["hit_rate"] < 40)
            else "medio" if row["picks"] >= min_muestra and (row["roi"] < 0 or row["hit_rate"] < 50)
            else "favorable" if row["picks"] >= min_muestra and row["roi"] > 8 and row["hit_rate"] >= 58
            else "neutro"
        ),
        axis=1,
    )

    return resumen.sort_values(by=["profit", "hit_rate"], ascending=[False, False]).reset_index(drop=True)


def _resumen_segmento_compuesto(df, columnas, nombre_columna, min_muestra=3):
    if df.empty or any(col not in df.columns for col in columnas):
        return pd.DataFrame()

    compuesto = df.copy()
    compuesto[nombre_columna] = compuesto.apply(
        lambda row: " | ".join(
            str(row.get(col) if pd.notna(row.get(col)) and str(row.get(col)).strip() else "Sin dato")
            for col in columnas
        ),
        axis=1,
    )
    return _resumen_segmento(compuesto, nombre_columna, min_muestra=min_muestra)


def _calcular_metricas_globales(cerradas):
    if cerradas.empty:
        return {"hit_rate": 0.5, "roi": 0.0, "avg_profit": 0.0}

    if "estado_prediccion" in cerradas.columns:
        lecturas = cerradas[cerradas["estado_prediccion"].isin(["Acierto", "Fallo"])]
        hit_rate = float((lecturas["estado_prediccion"] == "Acierto").mean()) if not lecturas.empty else 0.5
    else:
        hit_rate = float((cerradas["ganador"] == cerradas["resultado_real"]).mean())
    stake_columna = "stake_contable" if "stake_contable" in cerradas.columns else "stake_num"
    stakes = pd.to_numeric(cerradas[stake_columna], errors="coerce").replace(0, pd.NA).dropna()
    roi = float((cerradas["profit"].sum() / stakes.sum()) * 100) if not stakes.empty else 0.0
    avg_profit = float(cerradas["profit"].mean()) if "profit" in cerradas.columns else 0.0
    return {"hit_rate": hit_rate, "roi": roi, "avg_profit": avg_profit}


def _perfil_bayesiano(row, global_metrics, prior_strength=6):
    picks = int(row["picks"])
    aciertos = int(row["aciertos"])
    profit = float(row["profit"])

    posterior_hit = ((aciertos + (global_metrics["hit_rate"] * prior_strength)) / (picks + prior_strength)) if (picks + prior_strength) else global_metrics["hit_rate"]
    posterior_roi = ((row["roi"] * picks) + (global_metrics["roi"] * prior_strength)) / (picks + prior_strength) if (picks + prior_strength) else global_metrics["roi"]
    posterior_profit = (profit + (global_metrics["avg_profit"] * prior_strength)) / (picks + prior_strength) if (picks + prior_strength) else global_metrics["avg_profit"]

    if picks >= 4 and (posterior_roi < -12 or posterior_hit < 0.42):
        risk_level = "alto"
    elif picks >= 4 and (posterior_roi < -2 or posterior_hit < 0.5):
        risk_level = "medio"
    elif picks >= 4 and posterior_roi > 8 and posterior_hit >= 0.58:
        risk_level = "favorable"
    else:
        risk_level = "neutro"

    edge_delta = 0.0
    confianza_delta = 0
    stake_factor = 1.0

    if risk_level == "alto":
        edge_delta = 0.02
        confianza_delta = -8
        stake_factor = 0.7
    elif risk_level == "medio":
        edge_delta = 0.01
        confianza_delta = -4
        stake_factor = 0.85
    elif risk_level == "favorable":
        edge_delta = -0.005
        confianza_delta = 2
        stake_factor = 1.05

    return {
        "sample_size": picks,
        "hit_rate": round(posterior_hit * 100, 2),
        "roi": round(float(posterior_roi), 2),
        "avg_profit": round(float(posterior_profit), 3),
        "risk_level": risk_level,
        "edge_delta": edge_delta,
        "confianza_delta": confianza_delta,
        "stake_factor": stake_factor,
    }


def construir_perfiles_segmento(min_muestra=3):
    df = cargar_historial()
    cerradas = df[df["resultado_real"].notna()].copy()
    global_metrics = _calcular_metricas_globales(cerradas)

    perfiles = {
        "liga": {},
        "grupo_liga": {},
        "mercado": {},
        "liga_mercado": {},
        "grupo_liga_mercado": {},
        "fuente_cuotas": {},
        "prediccion_simple": {},
        "decision_simple": {},
        "ai_decision": {},
        "contexto_partido": {},
        "banda_confianza": {},
        "banda_edge": {},
        "banda_ai": {},
    }
    resumenes = {}

    if cerradas.empty:
        return perfiles, resumenes

    resumenes["liga_mercado"] = _resumen_segmento_compuesto(cerradas, ["liga", "mercado"], "liga_mercado", min_muestra=min_muestra)
    resumenes["grupo_liga_mercado"] = _resumen_segmento_compuesto(cerradas, ["grupo_liga", "mercado"], "grupo_liga_mercado", min_muestra=min_muestra)

    for clave_compuesto in ["liga_mercado", "grupo_liga_mercado"]:
        resumen = resumenes[clave_compuesto]
        if resumen.empty:
            continue
        for _, row in resumen.iterrows():
            if int(row["picks"]) < min_muestra:
                continue
            perfiles[clave_compuesto][row[clave_compuesto]] = _perfil_bayesiano(row, global_metrics, prior_strength=8)

    for columna, clave in [
        ("liga", "liga"),
        ("grupo_liga", "grupo_liga"),
        ("mercado", "mercado"),
        ("fuente_cuotas", "fuente_cuotas"),
        ("prediccion_simple", "prediccion_simple"),
        ("decision_simple", "decision_simple"),
        ("ai_decision", "ai_decision"),
        ("contexto_partido", "contexto_partido"),
        ("banda_confianza", "banda_confianza"),
        ("banda_edge", "banda_edge"),
        ("banda_ai", "banda_ai"),
    ]:
        resumen = _resumen_segmento(cerradas, columna, min_muestra=min_muestra)
        resumenes[clave] = resumen

        if resumen.empty:
            continue

        for _, row in resumen.iterrows():
            if int(row["picks"]) < min_muestra:
                continue
            perfiles[clave][row[columna]] = _perfil_bayesiano(row, global_metrics)

    return perfiles, resumenes


def calcular_calibracion_modelo():
    df = cargar_historial()
    cerradas = df[df["resultado_real"].notna()].copy()
    perfiles_segmento, _ = construir_perfiles_segmento()
    global_metrics = _calcular_metricas_globales(cerradas)

    if cerradas.empty or len(cerradas) < 8:
        return {
            "sample_size": int(len(cerradas)),
            "min_edge": 0.03,
            "confianza_bonus": 0,
            "stake_factor": 1.0,
            "max_stake_cap": 5,
            "segmentos": perfiles_segmento,
            "global_metrics": global_metrics,
        }

    lecturas_validas = cerradas[cerradas.get("estado_prediccion", "Sin lectura").isin(["Acierto", "Fallo"])].copy()
    aciertos = (
        (lecturas_validas["estado_prediccion"] == "Acierto").mean()
        if not lecturas_validas.empty
        else 0.5
    )
    roi, _ = calcular_roi()

    min_edge = 0.03
    confianza_bonus = 0
    stake_factor = 1.0
    max_stake_cap = 5

    if aciertos < 0.42:
        min_edge = 0.06
        confianza_bonus = -6
        stake_factor = 0.7
        max_stake_cap = 3
    elif aciertos < 0.5:
        min_edge = 0.05
        confianza_bonus = -3
        stake_factor = 0.85
        max_stake_cap = 4
    elif aciertos > 0.62 and roi > 0:
        min_edge = 0.025
        confianza_bonus = 3
        stake_factor = 1.05
        max_stake_cap = 5

    return {
        "sample_size": int(len(lecturas_validas)),
        "hit_rate": round(aciertos * 100, 2),
        "roi": roi,
        "min_edge": min_edge,
        "confianza_bonus": confianza_bonus,
        "stake_factor": stake_factor,
        "max_stake_cap": max_stake_cap,
        "segmentos": perfiles_segmento,
        "global_metrics": global_metrics,
    }


def resumen_historial():

    df = cargar_historial()

    if df.empty:
        return {
            "historial": df,
            "cerradas": 0,
            "pendientes": 0,
            "aciertos": 0,
            "fallos": 0,
            "hit_rate": 0.0,
            "profit_neto": 0.0,
        }

    cerradas = df[df["resultado_real"].notna()].copy()
    pendientes = int(df["resultado_real"].isna().sum())

    if cerradas.empty:
        return {
            "historial": df,
            "cerradas": 0,
            "pendientes": pendientes,
            "aciertos": 0,
            "fallos": 0,
            "hit_rate": 0.0,
            "profit_neto": 0.0,
        }

    apuestas_reales = cerradas[cerradas.get("es_apuesta_real", False)].copy()
    aciertos = int((apuestas_reales["estado_pick"] == "Acierto").sum())
    fallos = int((apuestas_reales["estado_pick"] == "Fallo").sum())
    hit_rate = round((aciertos / len(apuestas_reales)) * 100, 2) if len(apuestas_reales) else 0.0
    profit_neto = round(float(cerradas["profit"].sum()), 2) if "profit" in cerradas.columns else 0.0
    perfiles_segmento, resumenes_segmento = construir_perfiles_segmento()
    segmentos_decision = _resumen_segmento(cerradas, "decision_simple", min_muestra=1)
    segmentos_prediccion = _resumen_segmento(cerradas, "prediccion_simple", min_muestra=1)
    segmentos_contexto = _resumen_segmento(cerradas, "contexto_partido", min_muestra=1)
    segmentos_estado_consulta = _resumen_segmento(df.copy(), "estado_final_consulta", min_muestra=1)

    picks_apostar = (
        cerradas[cerradas["decision_simple"] == "Apostar"].copy()
        if "decision_simple" in cerradas.columns
        else pd.DataFrame()
    )
    picks_mirar = (
        cerradas[cerradas["decision_simple"] == "Mirar"].copy()
        if "decision_simple" in cerradas.columns
        else pd.DataFrame()
    )

    def _metricas_subgrupo(df_sub):
        if df_sub.empty:
            return {"picks": 0, "aciertos": 0, "hit_rate": 0.0, "profit": 0.0}
        aciertos_sub = int((df_sub["ganador"] == df_sub["resultado_real"]).sum())
        profit_sub = round(float(df_sub["profit"].sum()), 2) if "profit" in df_sub.columns else 0.0
        return {
            "picks": int(len(df_sub)),
            "aciertos": aciertos_sub,
            "hit_rate": round((aciertos_sub / len(df_sub)) * 100, 2),
            "profit": profit_sub,
        }

    top_ligas = resumenes_segmento.get("liga", pd.DataFrame())
    if not top_ligas.empty:
        top_ligas = top_ligas.sort_values(by=["profit", "hit_rate"], ascending=[False, False]).reset_index(drop=True)
    bottom_ligas = resumenes_segmento.get("liga", pd.DataFrame())
    if not bottom_ligas.empty:
        bottom_ligas = bottom_ligas.sort_values(by=["profit", "hit_rate"], ascending=[True, True]).reset_index(drop=True)

    pred_simple_acierto = 0.0
    if not cerradas.empty and "prediccion_simple" in cerradas.columns:
        pred_map = {
            "Gana el local": "Gana local",
            "Gana el visitante": "Gana visitante",
            "Empate": "Empate",
            "Partido para mirar": "No bet",
        }
        pred_real = cerradas["prediccion_simple"].map(pred_map).fillna("No bet")
        pred_simple_acierto = round(float((pred_real == cerradas["resultado_real"]).mean() * 100), 2)
    metricas_probabilisticas = _metricas_probabilisticas(cerradas)

    return {
        "historial": df,
        "cerradas": int(len(cerradas)),
        "pendientes": pendientes,
        "aciertos": aciertos,
        "fallos": fallos,
        "hit_rate": hit_rate,
        "profit_neto": profit_neto,
        "precision_prediccion_simple": pred_simple_acierto,
        "brier_score": metricas_probabilisticas["brier_score"],
        "log_loss": metricas_probabilisticas["log_loss"],
        "muestra_calibracion": metricas_probabilisticas["muestra"],
        "tabla_calibracion": metricas_probabilisticas["calibracion"],
        "max_drawdown": _max_drawdown(cerradas),
        "perfiles_segmento": perfiles_segmento,
        "segmentos_liga": resumenes_segmento.get("liga", pd.DataFrame()),
        "segmentos_mercado": resumenes_segmento.get("mercado", pd.DataFrame()),
        "segmentos_fuente": resumenes_segmento.get("fuente_cuotas", pd.DataFrame()),
        "segmentos_grupo_liga": resumenes_segmento.get("grupo_liga", pd.DataFrame()),
        "segmentos_liga_mercado": resumenes_segmento.get("liga_mercado", pd.DataFrame()),
        "segmentos_grupo_liga_mercado": resumenes_segmento.get("grupo_liga_mercado", pd.DataFrame()),
        "segmentos_confianza": resumenes_segmento.get("banda_confianza", pd.DataFrame()),
        "segmentos_edge": resumenes_segmento.get("banda_edge", pd.DataFrame()),
        "segmentos_ai": resumenes_segmento.get("banda_ai", pd.DataFrame()),
        "segmentos_decision": segmentos_decision,
        "segmentos_prediccion": segmentos_prediccion,
        "segmentos_contexto": segmentos_contexto,
        "segmentos_estado_consulta": segmentos_estado_consulta,
        "top_ligas": top_ligas.head(5) if not top_ligas.empty else pd.DataFrame(),
        "bottom_ligas": bottom_ligas.head(5) if not bottom_ligas.empty else pd.DataFrame(),
        "metricas_apostar": _metricas_subgrupo(picks_apostar),
        "metricas_mirar": _metricas_subgrupo(picks_mirar),
    }
