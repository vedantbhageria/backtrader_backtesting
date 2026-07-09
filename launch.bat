@echo off
rem ============================================================
rem  One-shot launcher: incremental data sync + backtest dashboard
rem
rem    launch.bat    sync bars (only downloads what postgres is missing),
rem                  then start the dashboard at http://localhost:8001
rem
rem  Bars and backtest results are stored in local PostgreSQL.
rem ============================================================
cd /d "%~dp0"

rem --- database connection (edit if your Postgres differs) ---
if not defined PGHOST     set PGHOST=localhost
if not defined PGPORT     set PGPORT=5432
if not defined PGUSER     set PGUSER=postgres
if not defined PGPASSWORD set PGPASSWORD=Forctix@0609
if not defined PGDATABASE set PGDATABASE=backtest

rem Always sync: the fetch is incremental — it asks postgres what's already
rem stored, downloads only the missing bars, then re-exports fresh CSVs.
echo [launch] syncing market data (incremental via postgres)...
python fetch_binance_csv.py

echo [launch] starting dashboard at http://localhost:8001
start "" http://localhost:8001
python server.py
