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

rem --- postgres binaries (edit if your install path/version differs) ---
set PGBIN=C:\Program Files\PostgreSQL\18\bin
set PGDATA_DIR=C:\Program Files\PostgreSQL\18\data

rem Make sure postgres is actually up and ACCEPTING CONNECTIONS before the
rem sync runs. Without this, a not-yet-started postgres makes fetch_binance_
rem csv.py's DB connect fail instantly, so it silently falls back to the slow
rem full-redownload path instead of the fast incremental one.
rem (Labels/goto must stay OUTSIDE any if/else (...) block -- cmd.exe reads a
rem parenthesized block as one unit and mishandles labels nested inside it.)
"%PGBIN%\pg_isready.exe" -h %PGHOST% -p %PGPORT% >nul 2>&1
if %errorlevel%==0 goto pgready

echo [launch] starting postgres...
rem log to %TEMP% -- the data\log dir under Program Files needs admin to write
"%PGBIN%\pg_ctl.exe" start -D "%PGDATA_DIR%" -l "%TEMP%\pg_startup.log"
set tries=0

:waitpg
"%PGBIN%\pg_isready.exe" -h %PGHOST% -p %PGPORT% >nul 2>&1
if %errorlevel%==0 goto pgready
set /a tries=%tries%+1
if %tries% GEQ 30 (
    echo [launch] WARNING: postgres not ready after 30s - continuing anyway
    goto pgready
)
timeout /t 1 >nul
goto waitpg

:pgready
echo [launch] postgres ready

rem No sync at startup -- use the dashboard's "Sync data" button (Config >
rem Data & modules) to refresh bars on demand.
echo [launch] skipping startup sync (use the dashboard Sync button)

rem Don't start a second server (it would just crash on "address already in
rem use") if one from an earlier launch.bat / manual run is still up.
netstat -ano | findstr /R /C:":8001 .*LISTENING" >nul
if %errorlevel%==0 (
    echo [launch] dashboard already running at http://localhost:8001 - reusing it
    start "" http://localhost:8001
) else (
    echo [launch] starting dashboard at http://localhost:8001
    start "" http://localhost:8001
    python server.py
)
