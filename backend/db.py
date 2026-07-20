

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import json
import os
import time

import psycopg2
from psycopg2.extras import Json, execute_values
# Start-Service postgresql-x64-18
DB = {
    'host': os.getenv('PGHOST', 'localhost'),
    'port': int(os.getenv('PGPORT', '5432')),
    'user': os.getenv('PGUSER', 'postgres'),
    'password': os.getenv('PGPASSWORD', 'Forctix@0609'),
}
DB_NAME = os.getenv('PGDATABASE', 'backtest')

# Hourly bars live in their own table (bars_1h) rather than an `interval`
# column on `bars`, so the existing 26M-row minute table needs no PK
# migration and 1h/1m never collide on a shared (symbol, ts) key.
_INTERVAL_TABLE = {'1m': 'bars', '1h': 'bars_1h'}


def _bars_table(interval):
    try:
        return _INTERVAL_TABLE[interval or '1m']
    except KeyError:
        raise ValueError('unsupported interval %r (want one of %s)'
                         % (interval, list(_INTERVAL_TABLE)))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol  text             NOT NULL,
    ts      timestamptz      NOT NULL,
    open    double precision NOT NULL,
    high    double precision NOT NULL,
    low     double precision NOT NULL,
    close   double precision NOT NULL,
    volume  double precision NOT NULL,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS bars_1h (
    symbol  text             NOT NULL,
    ts      timestamptz      NOT NULL,
    open    double precision NOT NULL,
    high    double precision NOT NULL,
    low     double precision NOT NULL,
    close   double precision NOT NULL,
    volume  double precision NOT NULL,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS runs (
    id            serial PRIMARY KEY,
    generated     timestamptz NOT NULL DEFAULT now(),
    run_tag       text,
    params        jsonb,
    summary       jsonb,
    pyfolio_stats jsonb
);
-- runs predating the run_tag column (added for durable artifact linking —
-- timestamp matching cross-linked parallel runs that finished microseconds
-- apart); NULL run_tag rows fall back to timestamp matching
ALTER TABLE runs ADD COLUMN IF NOT EXISTS run_tag text;

CREATE TABLE IF NOT EXISTS trades (
    run_id  integer REFERENCES runs(id) ON DELETE CASCADE,
    symbol  text NOT NULL,
    ts      timestamptz NOT NULL,
    pnl     double precision,
    pnlcomm double precision
);
CREATE INDEX IF NOT EXISTS trades_run_idx ON trades (run_id, symbol);

CREATE TABLE IF NOT EXISTS equity (
    run_id integer REFERENCES runs(id) ON DELETE CASCADE,
    ts     timestamptz NOT NULL,
    value  double precision NOT NULL
);
CREATE INDEX IF NOT EXISTS equity_run_idx ON equity (run_id, ts);

CREATE TABLE IF NOT EXISTS per_symbol (
    run_id integer REFERENCES runs(id) ON DELETE CASCADE,
    symbol text NOT NULL,
    trades integer,
    pnl    double precision,
    won    integer,
    PRIMARY KEY (run_id, symbol)
);
"""


def get_conn(_retries=3, _retry_delay=0.6):
    """Connect to Postgres, creating DB_NAME if it doesn't exist yet.

    On Windows the postgresql service can still be finishing its own startup
    (or hasn't been started at all) at the moment this app boots, so a
    'connection refused' here doesn't necessarily mean postgres is down for
    good — retry a few times with a short backoff before giving up. If it's
    genuinely not running, this still raises, and the caller's message should
    tell you to start it (Windows: `Start-Service postgresql-x64-18` from an
    elevated PowerShell, or set the service to Automatic in services.msc)."""
    last_err = None
    for attempt in range(_retries):
        try:
            return psycopg2.connect(dbname=DB_NAME, **DB)
        except psycopg2.OperationalError as e:
            if 'does not exist' in str(e):
                break   # DB missing (not a connectivity problem) -> create it below
            last_err = e
            if attempt < _retries - 1:
                time.sleep(_retry_delay)
    else:
        raise last_err
    # Target DB missing: connect to the maintenance DB and create it.
    admin = psycopg2.connect(dbname='postgres', **DB)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute('CREATE DATABASE "%s"' % DB_NAME)
    admin.close()
    return psycopg2.connect(dbname=DB_NAME, **DB)


def init_schema(conn):
    
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS public")
        cur.execute("SET search_path TO public")
        conn.commit()

        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            conn.commit()
            timescale = True
        except psycopg2.Error:
            conn.rollback()
            timescale = False

        cur.execute(_SCHEMA)
        conn.commit()

        if timescale:
            for table, col in (('bars', 'ts'), ('bars_1h', 'ts'), ('equity', 'ts')):
                try:
                    cur.execute(
                        "SELECT create_hypertable(%s, %s, "
                        "if_not_exists => TRUE, migrate_data => TRUE)",
                        (table, col))
                    conn.commit()
                except psycopg2.Error:
                    conn.rollback()  # plain table still works
    return timescale


def load_bars_csv(conn, symbol, csv_path):

    from datetime import datetime as _dt, timezone as _tz
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            ts = _dt.strptime(r['datetime'], '%Y-%m-%d %H:%M:%S').replace(
                tzinfo=_tz.utc, second=59, microsecond=999000)
            rows.append((symbol, ts, float(r['open']), float(r['high']),
                         float(r['low']), float(r['close']), float(r['volume'])))
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO bars (symbol, ts, open, high, low, close, volume) "
            "VALUES %s ON CONFLICT (symbol, ts) DO NOTHING",
            rows, page_size=2000)
    conn.commit()
    return len(rows)


def symbols(conn, interval='1m'):
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT symbol FROM %s ORDER BY symbol" % tbl)
        return [r[0] for r in cur.fetchall()]


def bars_span(conn, interval='1m'):

    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute("SELECT min(ts), max(ts) FROM %s" % tbl)
        return cur.fetchone()


def clear_run_history(conn):

    with conn.cursor() as cur:
        for tbl in ('trades', 'equity', 'per_symbol', 'runs'):
            cur.execute('DELETE FROM %s' % tbl)
    conn.commit()


def delete_run(conn, run_id):
    """Remove one run (and its trades/equity/per_symbol rows) from postgres."""
    with conn.cursor() as cur:
        for tbl in ('trades', 'equity', 'per_symbol'):
            cur.execute('DELETE FROM %s WHERE run_id=%%s' % tbl, (run_id,))
        cur.execute('DELETE FROM runs WHERE id=%s', (run_id,))
        deleted = cur.rowcount
    conn.commit()
    return deleted > 0


def per_symbol_span(conn, interval='1m'):

    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute("SELECT symbol, min(ts), max(ts), count(*) FROM %s "
                    "GROUP BY symbol ORDER BY symbol" % tbl)
        return cur.fetchall()


def coverage(conn, symbol, start, end, interval='1m'):

    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT min(ts), max(ts), count(*) FROM %s "
            "WHERE symbol=%%s AND ts >= %%s AND ts <= %%s" % tbl,
            (symbol, start, end))
        return cur.fetchone()


def insert_bars(conn, symbol, rows, interval='1m'):

    if not rows:
        return 0
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO %s (symbol, ts, open, high, low, close, volume) "
            "VALUES %%s ON CONFLICT (symbol, ts) DO UPDATE SET "
            "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
            "close=EXCLUDED.close, volume=EXCLUDED.volume" % tbl,
            [(symbol, r[0].replace(second=59, microsecond=999000)) + tuple(r[1:])
             for r in rows], page_size=10000)
    conn.commit()
    return len(rows)


def export_bars_csv(conn, symbol, start, end, path,
                    dt_format='%Y-%m-%d %H:%M:%S', interval='1m'):
 
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ts, open, high, low, close, volume FROM %s "
            "WHERE symbol=%%s AND ts >= %%s AND ts <= %%s ORDER BY ts" % tbl,
            (symbol, start, end))
        rows = cur.fetchall()
    if not rows:
        return 0
    from datetime import timezone as _tz
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['datetime', 'open', 'high', 'low', 'close', 'volume'])
        for ts, o, h, l, c, v in rows:
            w.writerow([ts.astimezone(_tz.utc).strftime(dt_format),
                        repr(o), repr(h), repr(l), repr(c), repr(v)])
    return len(rows)


def save_run(conn, results, trade_log, equity):
    """Persist one backtest run (results.json content + full trade/equity)."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO runs (generated, run_tag, params, summary, pyfolio_stats) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (results['generated'], results.get('run_tag'),
             Json(results['params']),
             Json(results['summary']),
             Json(results.get('pyfolio', {}).get('stats') or {})))
        run_id = cur.fetchone()[0]

        if trade_log:
            # `ts` = position close time (trade_log now uses exit_dt).
            execute_values(
                cur,
                "INSERT INTO trades (run_id, symbol, ts, pnl, pnlcomm) VALUES %s",
                [(run_id, t['symbol'], t.get('exit_dt') or t.get('dt'),
                  t['pnl'], t['pnlcomm'])
                 for t in trade_log], page_size=2000)

        if equity:
            execute_values(
                cur,
                "INSERT INTO equity (run_id, ts, value) VALUES %s",
                [(run_id, ts, v) for ts, v in equity], page_size=2000)

        per = results.get('per_symbol') or {}
        if per:
            execute_values(
                cur,
                "INSERT INTO per_symbol (run_id, symbol, trades, pnl, won) VALUES %s",
                [(run_id, s, v['trades'], v['pnl'], v['won'])
                 for s, v in per.items()])
    conn.commit()
    return run_id


def list_runs(conn, limit=50):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, generated, run_tag, params, summary FROM runs "
            "ORDER BY id DESC LIMIT %s", (limit,))
        return [{'id': i, 'generated': g.isoformat(), 'run_tag': rt,
                 'params': p, 'summary': s}
                for i, g, rt, p, s in cur.fetchall()]


if __name__ == '__main__':
    conn = get_conn()
    ts = init_schema(conn)
    print('schema ready (timescaledb=%s) at %s:%s/%s'
          % (ts, DB['host'], DB['port'], DB_NAME))
    print(json.dumps(list_runs(conn, 5), indent=1, default=str))
    conn.close()
