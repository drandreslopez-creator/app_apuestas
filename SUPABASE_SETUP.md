# Historial permanente con Supabase

1. Crea un proyecto gratuito en https://supabase.com.
2. Abre **SQL Editor**, crea una consulta nueva, pega el contenido de `supabase_schema.sql` y pulsa **Run**.
3. En **Project Settings > API**, copia `Project URL` y la clave `service_role`.
4. En Streamlit Cloud abre **Manage app > Settings > Secrets** y agrega:

```toml
SUPABASE_URL = "https://tu-proyecto.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "tu-clave-service-role"
```

5. Guarda. La app mostrará `Historial persistente` cuando la conexión funcione.

No subas la clave `service_role` a GitHub ni la compartas en capturas de pantalla.
