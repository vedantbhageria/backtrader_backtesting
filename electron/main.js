// Backtrader Dashboard — Electron wrapper.
//
// Use with Electron Fiddle: File > Open Fiddle > this folder, then Run.
// (Or later: `npx electron .` / export to electron-forge for a packaged app.)
//
// The dashboard is a thin client — everything comes from the FastAPI server
// (backend/server.py) over HTTP. This wrapper optionally STARTS that server
// (AUTO_START_SERVER below), waits for it to answer, then opens a window on
// it. Postgres must already be running (it usually is, as a resident
// process/service).

const { app, BrowserWindow, shell } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const http = require('http');
const path = require('path');

const DASH_URL = 'http://127.0.0.1:8001';
const AUTO_START_SERVER = true;           // set false if launch.bat runs it

// Windows toast notifications need a stable app identity.
app.setAppUserModelId('com.backtrader.dashboard');

// Project root: in dev/Fiddle it's the parent of electron/. In a PACKAGED
// exe __dirname points inside the app bundle, so read root.txt placed next
// to the exe (or the BACKTRADER_ROOT env var) instead.
function projectRoot() {
  if (process.env.BACKTRADER_ROOT) return process.env.BACKTRADER_ROOT;
  // Portable exes self-extract to %TEMP% — PORTABLE_EXECUTABLE_DIR is where
  // the REAL exe (and its root.txt) lives. Fall back to the exe dir (nsis/
  // unpacked builds), then to the repo layout (dev / Fiddle).
  const exeDir = process.env.PORTABLE_EXECUTABLE_DIR
              || path.dirname(process.execPath);
  try {
    const cfg = path.join(exeDir, 'root.txt');
    if (fs.existsSync(cfg)) return fs.readFileSync(cfg, 'utf8').trim();
  } catch (e) { /* fall through */ }
  return path.resolve(__dirname, '..');
}
const ROOT = projectRoot();

let serverProc = null;

function ping(url) {
  return new Promise(resolve => {
    const req = http.get(url, res => { res.resume(); resolve(true); });
    req.on('error', () => resolve(false));
    req.setTimeout(1500, () => { req.destroy(); resolve(false); });
  });
}

// Everything the launcher does is logged here — first place to look when the
// window opens on an error page.
const LOG = path.join(app.getPath('userData'), 'launcher.log');
function log(msg) {
  try { fs.appendFileSync(LOG, new Date().toISOString() + ' ' + msg + '\n'); }
  catch (e) { /* ignore */ }
}

function trySpawn(exe, script) {
  return new Promise(resolve => {
    let proc;
    try {
      proc = spawn(exe, [script], {cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe']});
    } catch (e) { log(`spawn ${exe} threw: ${e.message}`); return resolve(null); }
    let settled = false;
    proc.on('error', e => { log(`spawn ${exe} error: ${e.message}`);
                            if (!settled) { settled = true; resolve(null); } });
    proc.on('exit', c => { log(`server (${exe}) exited code=${c}`);
                           if (!settled) { settled = true; resolve(null); } });
    proc.stdout.on('data', d => log('[server] ' + String(d).trimEnd()));
    proc.stderr.on('data', d => log('[server:err] ' + String(d).trimEnd()));
    // if it's still alive after 3s, assume the interpreter started
    setTimeout(() => { if (!settled) { settled = true; resolve(proc); } }, 3000);
  });
}

async function ensureServer() {
  log(`--- launch: ROOT=${ROOT}`);
  if (await ping(DASH_URL + '/api/status')) { log('server already up'); return; }
  if (!AUTO_START_SERVER) return;
  const script = path.join(ROOT, 'backend', 'server.py');
  if (!fs.existsSync(script)) { log(`server script not found: ${script}`); return; }
  const candidates = [process.env.BACKTRADER_PYTHON, 'python', 'py'].filter(Boolean);
  for (const exe of candidates) {
    serverProc = await trySpawn(exe, script);
    if (serverProc) { log(`started server with ${exe} (pid ${serverProc.pid})`); break; }
  }
  if (!serverProc) { log('could not start the server with any interpreter'); return; }
  for (let i = 0; i < 30; i++) {                 // wait for it to answer
    if (await ping(DASH_URL + '/api/status')) { log('server is answering'); return; }
    await new Promise(r => setTimeout(r, 1000));
  }
  log('server never answered on ' + DASH_URL);
}

async function createWindow() {
  await ensureServer();
  const win = new BrowserWindow({
    width: 1500,
    height: 950,
    backgroundColor: '#0b0e14',
    autoHideMenuBar: true,
  });
  win.loadURL(DASH_URL);

  // "open report" links use target=_blank — send them to the default browser
  win.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  // Only kill the server if WE started it; a launch.bat-owned server stays up.
  if (serverProc) serverProc.kill();
  app.quit();
});
