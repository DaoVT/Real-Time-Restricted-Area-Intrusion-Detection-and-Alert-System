const { contextBridge } = require('electron');

// Renderer chỉ nhận surface nhỏ cần thiết, không chạm trực tiếp vào Node API.
contextBridge.exposeInMainWorld('intrusionApp', {
  platform: process.platform
});
