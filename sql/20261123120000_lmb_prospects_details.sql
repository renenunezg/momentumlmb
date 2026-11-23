BEGIN;

ALTER TABLE lmb.prospects
  ADD COLUMN IF NOT EXISTS details jsonb;

COMMENT ON COLUMN lmb.prospects.details IS 'Player bio, season stats, within-role percentiles, and model rank drivers for the expandable prospect card';

COMMIT;
