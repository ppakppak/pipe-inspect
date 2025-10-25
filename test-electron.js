const electron = require('electron');
console.log('module keys', Object.keys(electron));
console.log('app type', typeof electron.app);
console.log('has whenReady', electron.app && typeof electron.app.whenReady);
process.exit(0);
