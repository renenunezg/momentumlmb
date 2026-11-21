-- LMB player ratings in wins (backend/players). Latest season only: one row
-- per player who played it, with the WAR ledger and the forward grade for the
-- next season. About 1,000 rows (~0.16 MB with its index). History stays in
-- the local parquet build. Portraits come from the site's lmbPlayerHeadshotUrl
-- by player_id, so no image column.
CREATE TABLE lmb.player_ratings (
  player_id integer PRIMARY KEY,
  season smallint NOT NULL,
  name text NOT NULL,
  team_id integer,
  pos text NOT NULL CHECK (pos IN ('C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'DH', 'SP', 'RP')),
  age smallint,
  pa smallint NOT NULL,
  bf smallint NOT NULL,
  batting real NOT NULL,
  baserunning real NOT NULL,
  fielding real NOT NULL,
  positional real NOT NULL,
  pitching real NOT NULL,
  leverage real NOT NULL,
  replacement real NOT NULL,
  war real NOT NULL,
  grade_season smallint NOT NULL,
  grade smallint CHECK (grade BETWEEN 20 AND 80),
  grade_lo smallint,
  grade_hi smallint,
  proj_war real,
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- One row: the season's run scale, replacement level, positional scale,
-- grade weights and the validation the info tips quote.
CREATE TABLE lmb.player_rating_model (
  id smallint PRIMARY KEY DEFAULT 1 CHECK (id = 1),
  season smallint NOT NULL,
  model_version text NOT NULL,
  details jsonb NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['player_ratings', 'player_rating_model'] LOOP
    EXECUTE format('ALTER TABLE lmb.%I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('CREATE POLICY public_read ON lmb.%I FOR SELECT TO anon, authenticated USING (true)', t);
    EXECUTE format('GRANT SELECT ON lmb.%I TO anon, authenticated', t);
    EXECUTE format('GRANT ALL ON lmb.%I TO service_role', t);
    EXECUTE format('REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON lmb.%I FROM anon, authenticated', t);
    EXECUTE format('CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON lmb.%I '
                   'FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate()', t);
  END LOOP;
END $$;
