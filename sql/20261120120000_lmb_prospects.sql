BEGIN;

-- The LMB prospect board: young players with an LMB, LMP or minor league record, graded on the chance of an LMB
-- regular season (2.5 PA or 1.8 outs per team game) or an MLB debut within the next four LMB seasons. One row per
-- player and board season, replaced whole when the board is rescored; about 600 rows (well under 200 KB) a season.
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
  p_success numeric(5, 4) NOT NULL CHECK (p_success BETWEEN 0 AND 1),
  role_rank smallint NOT NULL,
  model_version text NOT NULL,
  scored_at timestamptz NOT NULL,
  PRIMARY KEY (snapshot_season, player_id)
);

ALTER TABLE lmb.prospects ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON lmb.prospects FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON lmb.prospects TO anon, authenticated;
GRANT ALL ON lmb.prospects TO service_role;

COMMIT;
