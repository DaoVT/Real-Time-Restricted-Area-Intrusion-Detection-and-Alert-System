const { app, BrowserWindow } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');

const isDev = process.env.VITE_DEV_SERVER_URL;
let backendProcess = null;

function startBackend() {
  if (process.env.DISABLE_ELECTRON_BACKEND === '1') return;
  backendProcess = spawn('python', ['-m', 'uvicorn', 'backend.app.main:app', '--host', '127.0.0.1', '--port', '8000'], {
    cwd: path.join(__dirname, '..'),
    windowsHide: true,
    stdio: 'ignore'
  });
}

function stopBackend() {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
  backendProcess = null;
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1280,
    height: 820,
    minWidth: 980,
    minHeight: 680,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  if (isDev) {
    win.loadURL(isDev);
  } else {
    win.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }
}

app.whenReady().then(() => {
  startBackend();
  createWindow();
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', stopBackend);
