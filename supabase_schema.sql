-- Run this once in Supabase: SQL Editor > New query > Run.
-- The app stores each historical snapshot as JSON so the schema remains
-- compatible when the prediction model evolves.
create table if not exists public.ai_predictor_history (
    unique_key text primary key,
    payload jsonb not null,
    updated_at timestamptz not null default now()
);

alter table public.ai_predictor_history enable row level security;

-- Streamlit uses the private service-role key stored in Secrets, so it can
-- write this table without exposing the key to visitors of the app.
