BEGIN;

-- Daily playoff odds: one row per team per date, the odds on the morning of as_of (games dated before it are
-- counted). Probabilities are basis points (0 to 10000) of reaching each stage: the top 6 of the zone, the zone
-- series (first-round winners and the best loser), the zone final, the Serie del Rey, and the title.
-- About 20 teams x 150 dates = 3,000 rows and 0.3 MB a season with the index; a rerun on unchanged results
-- writes nothing.
CREATE TABLE lmb.playoff_odds (
  season smallint NOT NULL,
  as_of date NOT NULL,
  team text NOT NULL,
  zone text NOT NULL CHECK (zone IN ('Norte', 'Sur')),
  wins smallint NOT NULL CHECK (wins >= 0),
  losses smallint NOT NULL CHECK (losses >= 0),
  proj_wins real NOT NULL,
  proj_losses real NOT NULL,
  p_playoffs smallint NOT NULL CHECK (p_playoffs BETWEEN 0 AND 10000),
  p_zone_series smallint NOT NULL CHECK (p_zone_series BETWEEN 0 AND p_playoffs),
  p_zone_final smallint NOT NULL CHECK (p_zone_final BETWEEN 0 AND p_zone_series),
  p_final smallint NOT NULL CHECK (p_final BETWEEN 0 AND p_zone_final),
  p_champion smallint NOT NULL CHECK (p_champion BETWEEN 0 AND p_final),
  PRIMARY KEY (season, as_of, team)
);

ALTER TABLE lmb.playoff_odds ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON lmb.playoff_odds FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON lmb.playoff_odds TO anon, authenticated;
GRANT ALL ON lmb.playoff_odds TO service_role;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON lmb.playoff_odds FROM anon, authenticated;

CREATE TRIGGER site_revalidate AFTER INSERT OR DELETE OR UPDATE OR TRUNCATE ON lmb.playoff_odds
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

COMMIT;
