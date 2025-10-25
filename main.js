const { app, BrowserWindow, dialog, ipcMain, protocol } = require('electron');
const path = require('path');

let mainWindow;

protocol.registerSchemesAsPrivileged([
  { scheme: 'http', privileges: { secure: false, standard: true, supportFetchAPI: true } },
  { scheme: 'https', privileges: { secure: true, standard: true, supportFetchAPI: true } }
]);

function createWindow() {
  const API_URL = process.env.API_URL || 'http://localhost:5003';

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    resizable: true,
    frame: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      additionalArguments: [`--api-url=${API_URL}`]
    },
    backgroundColor: '#1e1e1e',
    show: false
  });

  mainWindow.loadFile('index.html');

  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (input.key === 'F5') {
      mainWindow.webContents.reload();
    }
    if (input.control && input.shift && input.key === 'I') {
      mainWindow.webContents.toggleDevTools();
    }
    if (input.control && input.key === 'r') {
      mainWindow.webContents.reload();
    }
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  console.log('Window resizable:', mainWindow.isResizable());
}

function registerIpcHandlers() {
  if (ipcMain.listenerCount('dialog:openFile') > 0) {
    return;
  }
  ipcMain.handle('dialog:openFile', async (_event, options) => {
    if (!mainWindow) {
      throw new Error('Main window is not ready');
    }
    return dialog.showOpenDialog(mainWindow, options);
  });
}

app.whenReady().then(() => {
  registerIpcHandlers();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
