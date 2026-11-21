BEGIN;

-- The LMB prospect board: young players with an LMB, LMP or minor league record, ranked on their chance of an LMB
-- regular season (2.5 PA or 1.8 outs per team game) or an MLB debut within the next four LMB seasons. One row per
-- player and board season, replaced whole when the board is rescored; about 350 rows (about 50 KB) a season.
-- The site shows ranks only: role_rank orders hitters and pitchers separately. raw_score is the model output, kept
-- for context; it ranks well but is not a calibrated chance (on the 2021-2022 holdout it averaged .256 against
-- .147 observed for hitters and .205 against .081 for pitchers) and is never shown as a probability.
CREATE TABLE lmb.prospects (
  snapshot_season smallint NOT NULL,
  player_id integer NOT NULL,
  name text NOT NULL,
  role text NOT NULL CHECK (role IN ('H', 'P')),
  age numeric(3, 1) NOT NULL,
  team_id integer,
  team text,
  level text NOT NULL,
  mexican_born boolean NOT NULL,
  role_rank smallint NOT NULL CHECK (role_rank >= 1),
  raw_score numeric(5, 4) NOT NULL CHECK (raw_score BETWEEN 0 AND 1),
  model_version text NOT NULL,
  scored_at timestamptz NOT NULL,
  PRIMARY KEY (snapshot_season, player_id),
  UNIQUE (snapshot_season, role, role_rank)
);
COMMENT ON COLUMN lmb.prospects.raw_score IS 'Uncalibrated model output, context only; never display as a probability';

ALTER TABLE lmb.prospects ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON lmb.prospects FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON lmb.prospects TO anon, authenticated;
GRANT ALL ON lmb.prospects TO service_role;

-- The cached prospects page revalidates when the board changes, as for the other lmb tables.
CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON lmb.prospects
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
