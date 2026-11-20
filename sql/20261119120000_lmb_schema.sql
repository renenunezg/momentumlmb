BEGIN;

-- The LMB (Liga Mexicana de Beisbol) forecast schema. Shapes follow mlb's, minus the v1 archive, American odds,
-- Kelly stakes, run-line and totals columns: LMB picks are moneyline only, priced at the Kalshi yes ask plus fee.
-- The site reads with the anon key and writes nothing; the pipeline writes as service_role or postgres.

CREATE SCHEMA IF NOT EXISTS lmb;
GRANT USAGE ON SCHEMA lmb TO anon, authenticated, service_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA lmb GRANT SELECT ON TABLES TO anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA lmb GRANT ALL ON TABLES TO service_role;

CREATE FUNCTION lmb.refuse_truncate() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  RAISE EXCEPTION '% is append-only or frozen; TRUNCATE is refused', TG_TABLE_NAME;
END $$;

-- Games, keyed by Stats API gamePk. game_date is the official date (moves with a reschedule); start_time is UTC.
CREATE TABLE lmb.games (
  game_pk integer PRIMARY KEY,
  game_date date NOT NULL,
  start_time timestamptz,
  season integer NOT NULL,
  game_type text NOT NULL CHECK (game_type IN ('R', 'F', 'D', 'L', 'W')),
  home_team text NOT NULL,
  away_team text NOT NULL,
  home_team_id integer NOT NULL,
  away_team_id integer NOT NULL,
  venue_id integer NOT NULL,
  venue text,
  scheduled_innings smallint NOT NULL CHECK (scheduled_innings IN (7, 9)),
  status text NOT NULL,
  home_score smallint,
  away_score smallint,
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX games_date_idx ON lmb.games (game_date);

CREATE FUNCTION lmb.refuse_change() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  RAISE EXCEPTION '% is append-only (%)', TG_TABLE_NAME, TG_OP;
END $$;

-- Starters, lineups, workloads and the quote stream stay in the pipeline's own caches: the database holds only
-- what the site shows or grading needs (the shared project is near its size budget).

-- One closing Kalshi quote per game: the last quote the refresh job saw before first pitch. Write-once, pregame
-- only, on time.
CREATE TABLE lmb.closing_lines (
  game_pk integer PRIMARY KEY,
  season integer NOT NULL,
  start_time timestamptz NOT NULL,
  source text NOT NULL DEFAULT 'kalshi',
  fetched_at timestamptz NOT NULL,
  home_yes_bid numeric(5, 4),
  home_yes_ask numeric(5, 4),
  away_yes_bid numeric(5, 4),
  away_yes_ask numeric(5, 4)
);

CREATE FUNCTION lmb.refuse_closing_line_change() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF TG_OP <> 'INSERT' THEN
    RAISE EXCEPTION 'closing_lines is write-once (%)', TG_OP;
  END IF;
  IF NEW.fetched_at >= NEW.start_time THEN
    RAISE EXCEPTION 'Closing line for game % was fetched at %, not before its start %', NEW.game_pk, NEW.fetched_at, NEW.start_time;
  END IF;
  IF clock_timestamp() > NEW.start_time + interval '30 minutes' THEN
    RAISE EXCEPTION 'Closing line for game % arrived more than 30 minutes after its start %', NEW.game_pk, NEW.start_time;
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER refuse_closing_line_change BEFORE INSERT OR UPDATE OR DELETE ON lmb.closing_lines
  FOR EACH ROW EXECUTE FUNCTION lmb.refuse_closing_line_change();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON lmb.closing_lines
  FOR EACH STATEMENT EXECUTE FUNCTION lmb.refuse_truncate();

-- Forecasts: one row per team per game. model_outputs holds today's slate; model_outputs_season keeps every
-- game's forecast, frozen at first pitch. prediction_context carries the pick (market, team, probability, price,
-- fee, expected_value, edge, tier, policy) on both of a game's rows.
CREATE TABLE lmb.model_outputs (
  game_pk integer NOT NULL,
  team text NOT NULL,
  opponent text NOT NULL,
  is_home boolean NOT NULL,
  date date NOT NULL,
  start_time timestamptz,
  win_prob double precision NOT NULL,
  win_prob_p10 double precision,
  win_prob_p90 double precision,
  sim_win_prob double precision NOT NULL,
  market_prob double precision,
  kalshi_bid numeric(5, 4),
  kalshi_ask numeric(5, 4),
  expected_runs double precision NOT NULL,
  expected_runs_p10 double precision,
  expected_runs_p50 double precision,
  expected_runs_p90 double precision,
  runs_hist real[] NOT NULL,
  lineup_source text NOT NULL,
  lineup_hash text,
  prediction_updated_at timestamptz NOT NULL,
  prediction_context jsonb NOT NULL DEFAULT '{}'::jsonb,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (game_pk, team)
);
CREATE TABLE lmb.model_outputs_season (LIKE lmb.model_outputs INCLUDING ALL);
CREATE INDEX model_outputs_season_date_idx ON lmb.model_outputs_season (date);

-- A forecast cannot change once its game has started (or when its first pitch is unknown), whatever the write path.
CREATE FUNCTION lmb.refuse_started_forecast_write() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
DECLARE
  target integer := CASE WHEN TG_OP = 'DELETE' THEN OLD.game_pk ELSE NEW.game_pk END;
  game record;
BEGIN
  SELECT start_time, status INTO game FROM lmb.games WHERE game_pk = target;
  IF NOT FOUND OR game.start_time IS NULL OR game.start_time <= clock_timestamp()
     OR game.status IN ('Live', 'In Progress', 'Final', 'Completed Early', 'Game Over') THEN
    RAISE EXCEPTION 'Forecast for game % is frozen: first pitch passed or unknown (%)', target, TG_OP;
  END IF;
  IF TG_OP = 'UPDATE' AND OLD.game_pk IS DISTINCT FROM NEW.game_pk THEN
    RAISE EXCEPTION 'Forecast rows cannot move between games';
  END IF;
  RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
END $$;
CREATE TRIGGER refuse_started_forecast_write BEFORE INSERT OR UPDATE OR DELETE ON lmb.model_outputs_season
  FOR EACH ROW EXECUTE FUNCTION lmb.refuse_started_forecast_write();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON lmb.model_outputs_season
  FOR EACH STATEMENT EXECUTE FUNCTION lmb.refuse_truncate();

-- Compact append-only history of the pick: a row when a game is first published and whenever its pick (team,
-- tier or price) changes before first pitch. The frozen season row is the full record; this shows how the pick
-- moved. About 60 bytes a row, one or two rows a game.
CREATE TABLE lmb.forecast_history (
  game_pk integer NOT NULL,
  team text NOT NULL,
  published_at timestamptz NOT NULL,
  win_prob real NOT NULL,
  pick_team text,
  pick_tier text,
  pick_price real
);
CREATE INDEX forecast_history_game_idx ON lmb.forecast_history (game_pk);

CREATE FUNCTION lmb.record_forecast_history() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF NOT NEW.is_home THEN
    RETURN NEW;
  END IF;
  IF TG_OP = 'INSERT'
     OR (NEW.prediction_context #>> '{pick,team}', NEW.prediction_context #>> '{pick,tier}', NEW.prediction_context #>> '{pick,price}')
        IS DISTINCT FROM
        (OLD.prediction_context #>> '{pick,team}', OLD.prediction_context #>> '{pick,tier}', OLD.prediction_context #>> '{pick,price}') THEN
    INSERT INTO lmb.forecast_history (game_pk, team, published_at, win_prob, pick_team, pick_tier, pick_price)
    VALUES (NEW.game_pk, NEW.team, NEW.prediction_updated_at, NEW.win_prob, NEW.prediction_context #>> '{pick,team}',
            NEW.prediction_context #>> '{pick,tier}', (NEW.prediction_context #>> '{pick,price}')::real);
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER record_forecast_history AFTER INSERT OR UPDATE ON lmb.model_outputs_season
  FOR EACH ROW EXECUTE FUNCTION lmb.record_forecast_history();
CREATE TRIGGER refuse_change BEFORE UPDATE OR DELETE ON lmb.forecast_history
  FOR EACH ROW EXECUTE FUNCTION lmb.refuse_change();

-- The daily minimum slate: ranked once per date, 60 minutes before the date's first scheduled first pitch.
CREATE TABLE lmb.pick_slates (
  date date PRIMARY KEY,
  ranked_at timestamptz NOT NULL,
  game_pks integer[] NOT NULL
);
CREATE FUNCTION lmb.refuse_slate_change() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  RAISE EXCEPTION 'pick_slates is write-once (%)', TG_OP;
END $$;
CREATE TRIGGER refuse_slate_change BEFORE UPDATE OR DELETE ON lmb.pick_slates
  FOR EACH ROW EXECUTE FUNCTION lmb.refuse_slate_change();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON lmb.pick_slates
  FOR EACH STATEMENT EXECUTE FUNCTION lmb.refuse_truncate();

-- Nightly evaluation. The Kalshi columns cover only games with a closing quote; never compare them with the
-- all-games columns as one population.
CREATE TABLE lmb.model_evaluation (
  "window" text PRIMARY KEY CHECK ("window" IN ('day', '7d', '30d', 'season')),
  as_of date NOT NULL,
  season integer NOT NULL,
  n_games integer NOT NULL,
  log_loss double precision,
  brier double precision,
  accuracy double precision,
  kalshi_games integer NOT NULL DEFAULT 0,
  model_log_loss_on_kalshi double precision,
  kalshi_log_loss double precision,
  model_brier_on_kalshi double precision,
  kalshi_brier double precision,
  pick_count integer NOT NULL DEFAULT 0,
  pick_wins integer NOT NULL DEFAULT 0,
  pick_units double precision NOT NULL DEFAULT 0,
  expected_runs_mae double precision,
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE lmb.model_calibration (
  season integer NOT NULL,
  bin smallint NOT NULL,
  n integer NOT NULL,
  mean_pred double precision NOT NULL,
  mean_obs double precision NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (season, bin)
);

-- One row per graded pick: 1u at the frozen Kalshi ask plus fee.
CREATE VIEW lmb.bet_ledger_v WITH (security_invoker = true) AS
SELECT m.game_pk,
       m.date,
       g.season,
       m.prediction_context -> 'pick' ->> 'team' AS team,
       m.prediction_context -> 'pick' ->> 'tier' AS tier,
       m.prediction_context -> 'pick' ->> 'policy' AS policy,
       (m.prediction_context -> 'pick' ->> 'probability')::double precision AS probability,
       (m.prediction_context -> 'pick' ->> 'price')::double precision AS price,
       (m.prediction_context -> 'pick' ->> 'fee')::double precision AS fee,
       (m.prediction_context -> 'pick' ->> 'expected_value')::double precision AS expected_value,
       g.status,
       CASE
         WHEN g.status NOT IN ('Final', 'Completed Early', 'Game Over') OR g.home_score IS NULL THEN NULL
         WHEN g.home_score = g.away_score THEN NULL
         ELSE ((g.home_score > g.away_score) = ((m.prediction_context -> 'pick' ->> 'team') = g.home_team))
       END AS won,
       CASE
         WHEN g.status NOT IN ('Final', 'Completed Early', 'Game Over') OR g.home_score IS NULL OR g.home_score = g.away_score THEN NULL
         WHEN (g.home_score > g.away_score) = ((m.prediction_context -> 'pick' ->> 'team') = g.home_team)
           THEN 1.0 / ((m.prediction_context -> 'pick' ->> 'price')::double precision
                       + (m.prediction_context -> 'pick' ->> 'fee')::double precision) - 1.0
         ELSE -1.0
       END AS units
FROM lmb.model_outputs_season m
JOIN lmb.games g ON g.game_pk = m.game_pk AND g.game_date = m.date
WHERE m.is_home AND m.prediction_context ? 'pick';

-- Read-only for the browser key.
DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['games', 'closing_lines',
                           'model_outputs', 'model_outputs_season', 'forecast_history', 'pick_slates',
                           'model_evaluation', 'model_calibration'] LOOP
    EXECUTE format('ALTER TABLE lmb.%I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('CREATE POLICY public_read ON lmb.%I FOR SELECT TO anon, authenticated USING (true)', t);
  END LOOP;
END $$;
GRANT SELECT ON ALL TABLES IN SCHEMA lmb TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA lmb TO service_role;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA lmb FROM anon, authenticated;

-- Pages that are cached revalidate when their data changes. The games page is dynamic and reloads on updated_at.
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON lmb.model_evaluation
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON lmb.model_calibration
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON lmb.model_outputs_season
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
