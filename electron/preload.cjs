const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('intrusionApp', {
  platform: process.platform
});
