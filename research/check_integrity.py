"""Scan the bars table and report every break in the timestamp sequence.

For each symbol it walks the ordered timestamps, subtracts successive ones, and
flags anything that isn't exactly one bar apart:

    gap      diff > interval   -> bars are missing between the two timestamps
    overlap  diff < interval   -> rows closer than one bar (duplicate minutes,
                                  sub-second timestamp variants, bad inserts)

Reports, per symbol, the time range of every break plus a count of compromised
points, and a portfolio-wide summary at the end.

Usage:
    python check_integrity.py                     # all symbols, 1-minute bars
    python check_integrity.py --symbol BTCUSDT    # one symbol
    python check_integrity.py --interval 60       # bar size in seconds
    python check_integrity.py --limit 20          # detail rows shown per symbol
    python check_integrity.py --csv breaks.csv    # dump every break to CSV

Exit code is 1 when any break is found, so it can gate a pipeline.
"""


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, 'backend'), os.path.join(_ROOT, 'strategies')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import csv
import sys

import db

MAX_DETAIL = 200_000   # per-symbol break rows we're willing to pull for listing


def _fmt(n):
    return format(n, ',')


def summary_rows(conn, interval, symbol=None):
    """Per-symbol break counts, computed entirely in SQL (LAG window)."""
    where = 'WHERE symbol = %s' if symbol else ''
    args = ([symbol] if symbol else []) + [interval] * 5
    sql = """
    WITH d AS (
        SELECT symbol, ts,
               LAG(ts) OVER (PARTITION BY symbol ORDER BY ts) AS prev_ts
        FROM bars {where}
    ), a AS (
        SELECT symbol, prev_ts, ts,
               EXTRACT(EPOCH FROM (ts - prev_ts)) AS gap
        FROM d
        WHERE prev_ts IS NOT NULL
          AND EXTRACT(EPOCH FROM (ts - prev_ts)) <> %s
    )
    SELECT symbol,
           COUNT(*)                                            AS breaks,
           COUNT(*) FILTER (WHERE gap > %s)                    AS gaps,
           COUNT(*) FILTER (WHERE gap < %s)                    AS overlaps,
           COALESCE(SUM(CASE WHEN gap > %s
                             THEN (gap / %s)::bigint - 1 ELSE 0 END), 0) AS missing_bars,
           MIN(prev_ts) AS first_bad,
           MAX(ts)      AS last_bad
    FROM a GROUP BY symbol ORDER BY symbol
    """.format(where=where)
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def detail_rows(conn, interval, symbol, limit=None):
    """The individual breaks for one symbol, oldest first."""
    sql = """
    WITH d AS (
        SELECT ts, LAG(ts) OVER (ORDER BY ts) AS prev_ts
        FROM bars WHERE symbol = %s
    )
    SELECT prev_ts, ts, EXTRACT(EPOCH FROM (ts - prev_ts)) AS gap
    FROM d
    WHERE prev_ts IS NOT NULL
      AND EXTRACT(EPOCH FROM (ts - prev_ts)) <> %s
    ORDER BY ts
    """
    if limit:
        sql += ' LIMIT %d' % int(limit)
    with conn.cursor() as cur:
        cur.execute(sql, (symbol, interval))
        return cur.fetchall()


def coverage_rows(conn, symbol=None):
    where = 'WHERE symbol = %s' if symbol else ''
    with conn.cursor() as cur:
        cur.execute(
            'SELECT symbol, COUNT(*), MIN(ts), MAX(ts) FROM bars %s '
            'GROUP BY symbol ORDER BY symbol' % where,
            ([symbol] if symbol else []))
        return {r[0]: (r[1], r[2], r[3]) for r in cur.fetchall()}


def main():
    p = argparse.ArgumentParser(description='Check bar timestamp continuity.')
    p.add_argument('--interval', type=int, default=60,
                   help='expected seconds between bars (default 60 = 1m bars)')
    p.add_argument('--symbol', default=None, help='check a single symbol')
    p.add_argument('--limit', type=int, default=10,
                   help='max break rows printed per symbol (0 = all)')
    p.add_argument('--csv', default=None, help='write every break to this CSV')
    args = p.parse_args()

    conn = db.get_conn()
    cov = coverage_rows(conn, args.symbol)
    if not cov:
        print('No bars found%s.' % (' for %s' % args.symbol if args.symbol else ''))
        return 0

    print('Data integrity check - expected interval %ds (%.0f-minute bars)'
          % (args.interval, args.interval / 60.0))
    total_bars = sum(v[0] for v in cov.values())
    print('Scanned %d symbol(s), %s bars\n' % (len(cov), _fmt(total_bars)))

    breaks = {r[0]: r for r in summary_rows(conn, args.interval, args.symbol)}

    csv_w = csv_f = None
    if args.csv:
        csv_f = open(args.csv, 'w', newline='', encoding='utf-8')
        csv_w = csv.writer(csv_f)
        csv_w.writerow(['symbol', 'kind', 'from_ts', 'to_ts',
                        'gap_seconds', 'bars_missing'])

    clean, total_breaks, total_missing = [], 0, 0
    for sym in sorted(cov):
        nbars, first, last = cov[sym]
        if sym not in breaks:
            clean.append(sym)
            continue

        _, nbreak, ngap, nover, missing, first_bad, last_bad = breaks[sym]
        total_breaks += nbreak
        total_missing += int(missing)

        print('%s  --  %s bars, %s .. %s' % (sym, _fmt(nbars), first, last))
        print('   %d break(s): %d gap(s), %d overlap(s); %s bar(s) missing'
              % (nbreak, ngap, nover, _fmt(int(missing))))
        print('   compromised span: %s .. %s' % (first_bad, last_bad))

        # A wrong --interval makes every row a "break"; don't pull millions.
        if nbreak > MAX_DETAIL:
            print('     (%s breaks - too many to list; is --interval right?)\n'
                  % _fmt(nbreak))
            continue

        # Fetch all breaks once: print the first `limit`, write them all to CSV.
        rows = detail_rows(conn, args.interval, sym)
        shown = rows if args.limit == 0 else rows[:args.limit]
        for i, (prev_ts, ts, gap) in enumerate(rows):
            gap = int(gap)
            if gap > args.interval:
                kind = 'GAP'
                miss = gap // args.interval - 1
                note = '%s bar(s) missing' % _fmt(miss)
            else:
                kind = 'OVERLAP'
                miss = 0
                note = 'rows %ds apart (< one bar)' % gap
            if i < len(shown):
                print('     %-8s %s -> %s  (%ss, %s)'
                      % (kind, prev_ts, ts, _fmt(gap), note))
            if csv_w:
                csv_w.writerow([sym, kind, prev_ts, ts, gap, miss])
        if len(shown) < len(rows):
            print('     ... and %d more (use --limit 0 to see all)'
                  % (len(rows) - len(shown)))
        print()

    if csv_f:
        csv_f.close()

    print('=' * 62)
    print('clean symbols     : %d / %d' % (len(clean), len(cov)))
    print('symbols w/ breaks : %d' % len(breaks))
    print('total breaks      : %s' % _fmt(total_breaks))
    print('total bars missing: %s' % _fmt(total_missing))
    if args.csv:
        print('breaks written to : %s' % args.csv)
    conn.close()
    return 1 if breaks else 0


if __name__ == '__main__':
    sys.exit(main())
