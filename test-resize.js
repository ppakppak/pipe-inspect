const { app, BrowserWindow } = require('electron');

app.whenReady().then(() => {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    resizable: true,
    frame: true
  });
  
  win.loadURL('data:text/html,<h1>Resize Test - Try dragging corners</h1>');
  
  console.log('Resizable:', win.isResizable());
  console.log('Maximizable:', win.isMaximizable());
  console.log('Minimizable:', win.isMinimizable());
});

app.on('window-all-closed', () => {
  app.quit();
});
