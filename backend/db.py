

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import csv
import json
import os

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
    params        jsonb,
    summary       jsonb,
    pyfolio_stats jsonb
);

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


def get_conn():
    """Connect to PGDATABASE, creating it if it doesn't exist yet."""
    try:
        return psycopg2.connect(dbname=DB_NAME, **DB)
    except psycopg2.OperationalError as e:
        if 'does not exist' not in str(e):
            raise
    # Target DB missing: connect to the maintenance DB and create it.
    admin = psycopg2.connect(dbname='postgres', **DB)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute('CREATE DATABASE "%s"' % DB_NAME)
    admin.close()
    return psycopg2.connect(dbname=DB_NAME, **DB)


def init_schema(conn):
    """Create tables; make bars/equity hypertables when TimescaleDB is present."""
    with conn.cursor() as cur:
        # Ensure a schema exists to create tables in. Clearing the DB with
        # `DROP SCHEMA public CASCADE` leaves no schema in the search_path,
        # which otherwise fails with "no schema has been selected to create in".
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
    """Idempotent bulk-load of one symbol's OHLCV CSV into bars.

    Timestamps are canonicalized to the minute-aligned close (:59.999) — the
    same convention insert_bars uses — so CSV loads (which drop milliseconds)
    can't create :59.000 / :59.999 duplicate rows against direct inserts.
    """
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
    """(min_ts, max_ts) across all stored bars of this interval."""
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute("SELECT min(ts), max(ts) FROM %s" % tbl)
        return cur.fetchone()


def clear_run_history(conn):
    """Delete every recorded run (runs + its trades/equity/per_symbol rows).
    Bar data is untouched."""
    with conn.cursor() as cur:
        for tbl in ('trades', 'equity', 'per_symbol', 'runs'):
            cur.execute('DELETE FROM %s' % tbl)
    conn.commit()


def per_symbol_span(conn, interval='1m'):
    """[(symbol, min_ts, max_ts, count)] for every symbol stored at `interval`.
    One GROUP BY pass — powers the dashboard's latest-bar / integrity views."""
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute("SELECT symbol, min(ts), max(ts), count(*) FROM %s "
                    "GROUP BY symbol ORDER BY symbol" % tbl)
        return cur.fetchall()


def coverage(conn, symbol, start, end, interval='1m'):
    """(min_ts, max_ts, count) of stored bars for symbol within [start, end]."""
    tbl = _bars_table(interval)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT min(ts), max(ts), count(*) FROM %s "
            "WHERE symbol=%%s AND ts >= %%s AND ts <= %%s" % tbl,
            (symbol, start, end))
        return cur.fetchone()


def insert_bars(conn, symbol, rows, interval='1m'):
    """Upsert (ts, o, h, l, c, v) rows; refetched candles overwrite partials.

    Timestamps are canonicalized to the minute-aligned close (:59.999, the
    Binance 1m convention) so bars from different sources — vision ZIPs, REST,
    or millisecond-truncated CSV loads — collapse onto one row per minute
    instead of leaving :59.000 / :59.999 duplicates.
    """
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
    """Write one symbol's bars in [start, end] to a backtrader-ready CSV."""
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
            "INSERT INTO runs (generated, params, summary, pyfolio_stats) "
            "VALUES (%s, %s, %s, %s) RETURNING id",
            (results['generated'], Json(results['params']),
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
            "SELECT id, generated, params, summary FROM runs "
            "ORDER BY id DESC LIMIT %s", (limit,))
        return [{'id': i, 'generated': g.isoformat(),
                 'params': p, 'summary': s}
                for i, g, p, s in cur.fetchall()]


if __name__ == '__main__':
    conn = get_conn()
    ts = init_schema(conn)
    print('schema ready (timescaledb=%s) at %s:%s/%s'
          % (ts, DB['host'], DB['port'], DB_NAME))
    print(json.dumps(list_runs(conn, 5), indent=1, default=str))
    conn.close()
