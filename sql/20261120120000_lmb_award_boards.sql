BEGIN;

-- LMB award forecasts, frozen when published: the top 10 of each race (MVP, POY pitcher, REL reliever, ROY rookie,
-- MOY manager; player_id is the manager's person ID for MOY). rank is null only for a winner who was off the board,
-- added by the grader. win_probability is null where the award earned a ranking only. won is null until graded.
-- About 50 rows (10 kB) and 5 model rows (15 kB) a season.
CREATE TABLE lmb.award_boards (
  season smallint NOT NULL,
  award text NOT NULL CHECK (award IN ('MVP', 'POY', 'REL', 'ROY', 'MOY')),
  player_id integer NOT NULL,
  player_name text NOT NULL,
  team text,
  rank smallint,
  win_probability real,
  stat_line text NOT NULL DEFAULT '',
  won boolean,
  PRIMARY KEY (season, award, player_id)
);
-- One row per award and season: when the board froze, whether the league had already announced its winners,
-- and the leave-one-year-out validation behind it.
CREATE TABLE lmb.award_models (
  season smallint NOT NULL,
  award text NOT NULL CHECK (award IN ('MVP', 'POY', 'REL', 'ROY', 'MOY')),
  as_of timestamptz NOT NULL,
  model_version text NOT NULL,
  after_announcement boolean NOT NULL,
  backtest jsonb NOT NULL,
  PRIMARY KEY (season, award)
);
ALTER TABLE lmb.award_boards ENABLE ROW LEVEL SECURITY;
ALTER TABLE lmb.award_models ENABLE ROW LEVEL SECURITY;
CREATE POLICY public_read ON lmb.award_boards FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY public_read ON lmb.award_models FOR SELECT TO anon, authenticated USING (true);
GRANT SELECT ON lmb.award_boards, lmb.award_models TO anon, authenticated;
GRANT ALL ON lmb.award_boards, lmb.award_models TO service_role;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON lmb.award_boards, lmb.award_models
  FROM anon, authenticated;

-- A forecast cannot move after it is published. The grader may only set won, once, on a row that has none.
CREATE FUNCTION lmb.refuse_award_board_change() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'award_boards is frozen (DELETE)';
  END IF;
  IF OLD.won IS NOT NULL OR (to_jsonb(NEW) - 'won') <> (to_jsonb(OLD) - 'won') THEN
    RAISE EXCEPTION 'award_boards is frozen: only an ungraded row can take won';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER refuse_award_board_change BEFORE UPDATE OR DELETE ON lmb.award_boards
  FOR EACH ROW EXECUTE FUNCTION lmb.refuse_award_board_change();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON lmb.award_boards
  FOR EACH STATEMENT EXECUTE FUNCTION lmb.refuse_truncate();
CREATE TRIGGER refuse_change BEFORE UPDATE OR DELETE ON lmb.award_models
  FOR EACH ROW EXECUTE FUNCTION lmb.refuse_change();
CREATE TRIGGER refuse_truncate BEFORE TRUNCATE ON lmb.award_models
  FOR EACH STATEMENT EXECUTE FUNCTION lmb.refuse_truncate();

CREATE TRIGGER site_revalidate AFTER INSERT OR UPDATE OR DELETE ON lmb.award_boards
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();
CREATE TRIGGER site_revalidate AFTER INSERT OR UPDATE OR DELETE ON lmb.award_models
  FOR EACH STATEMENT EXECUTE FUNCTION public.site_revalidate();

NOTIFY pgrst, 'reload schema';

COMMIT;
