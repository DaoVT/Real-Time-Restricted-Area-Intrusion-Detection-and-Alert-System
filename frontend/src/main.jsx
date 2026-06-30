import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Play, RotateCcw, Square, Upload } from 'lucide-react';
import './styles.css';

const API_BASE = 'http://127.0.0.1:8000';
const WS_BASE = 'ws://127.0.0.1:8000';

function VideoPanel({ file, previewUrl, liveFrameUrl, points, onAddPoint, onClearPoints }) {
  const imageRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const draw = () => {
      canvas.width = image.clientWidth || 640;
      canvas.height = image.clientHeight || 360;
      const context = canvas.getContext('2d');
      context.clearRect(0, 0, canvas.width, canvas.height);
      if (points.length > 0) {
        context.strokeStyle = '#f59e0b';
        context.fillStyle = 'rgba(245, 158, 11, 0.18)';
        context.lineWidth = 2;
        context.beginPath();
        points.forEach((point, index) => {
          const x = point.x * canvas.width;
          const y = point.y * canvas.height;
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        if (points.length >= 3) context.closePath();
        context.stroke();
        if (points.length >= 3) context.fill();
        points.forEach((point) => {
          context.beginPath();
          context.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
          context.fillStyle = '#ef4444';
          context.fill();
        });
      }
    };
    draw();
    window.addEventListener('resize', draw);
    return () => window.removeEventListener('resize', draw);
  }, [points, previewUrl, liveFrameUrl]);

  const handleCanvasClick = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    onAddPoint({
      x: Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1),
      y: Math.min(Math.max((event.clientY - rect.top) / rect.height, 0), 1)
    });
  };

  return (
    <section className="video-panel">
      <div className="panel-title">
        <h2>Video</h2>
        <button type="button" className="icon-button" onClick={onClearPoints} title="Clear polygon">
          <RotateCcw size={18} />
        </button>
      </div>
      <div className="video-stage">
        {(liveFrameUrl || previewUrl) ? (
          <img ref={imageRef} src={liveFrameUrl || previewUrl} alt={file?.name || 'Video preview'} />
        ) : (
          <div className="empty-stage">Select a video</div>
        )}
        <canvas ref={canvasRef} onClick={handleCanvasClick} />
      </div>
    </section>
  );
}

function ControlPanel({ onFileChange, onStart, onStop, onReset, disabled, status, backendStatus, pointsCount }) {
  return (
    <section className="control-panel">
      <h2>Controls</h2>
      <label className="file-picker">
        <Upload size={18} />
        <span>Choose video</span>
        <input type="file" accept="video/*" onChange={(event) => onFileChange(event.target.files?.[0] || null)} />
      </label>
      <div className="button-row">
        <button type="button" onClick={onStart} disabled={disabled || pointsCount < 3}>
          <Play size={16} /> Start
        </button>
        <button type="button" onClick={onStop} disabled={!status || status === 'COMPLETED' || status === 'STOPPED'}>
          <Square size={16} /> Stop
        </button>
        <button type="button" onClick={onReset}>
          <RotateCcw size={16} /> Reset
        </button>
      </div>
      <div className="status-strip">
        <span>Backend</span>
        <strong className={backendStatus === 'connected' ? 'good' : 'bad'}>{backendStatus}</strong>
      </div>
      <div className="status-strip">
        <span>Job</span>
        <strong>{status || 'idle'}</strong>
      </div>
    </section>
  );
}

function StatisticsPanel({ stats }) {
  const items = [
    ['Progress', `${Math.round((stats.progress || 0) * 100)}%`],
    ['FPS', stats.fps.toFixed(1)],
    ['People', stats.people_count],
    ['Inside', stats.inside_count],
    ['Intrusions', stats.intrusion_count]
  ];
  return (
    <section className="statistics-panel">
      <h2>Statistics</h2>
      <div className="stat-grid">
        {items.map(([label, value]) => (
          <div className="stat-tile" key={label}>
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function EventTable({ events }) {
  return (
    <section className="event-panel">
      <h2>Events</h2>
      <table>
        <thead>
          <tr>
            <th>Frame</th>
            <th>Track</th>
            <th>Confidence</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          {events.map((event, index) => (
            <tr key={`${event.frame}-${event.track_id}-${index}`}>
              <td>{event.frame}</td>
              <td>{event.track_id}</td>
              <td>{Number(event.confidence || 0).toFixed(2)}</td>
              <td>{event.evidence_path ? <span title={event.evidence_path}>saved</span> : '-'}</td>
            </tr>
          ))}
          {events.length === 0 && (
            <tr>
              <td colSpan="4" className="empty-table">No intrusion events</td>
            </tr>
          )}
        </tbody>
      </table>
    </section>
  );
}

function useBackendHealth() {
  const [status, setStatus] = useState('checking');
  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        const response = await fetch(`${API_BASE}/health`);
        if (active) setStatus(response.ok ? 'connected' : 'disconnected');
      } catch {
        if (active) setStatus('disconnected');
      }
    };
    check();
    const id = window.setInterval(check, 5000);
    return () => {
      active = false;
      window.clearInterval(id);
    };
  }, []);
  return status;
}

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [videoUrl, setVideoUrl] = useState('');
  const [liveFrameUrl, setLiveFrameUrl] = useState('');
  const [points, setPoints] = useState([]);
  const [jobId, setJobId] = useState('');
  const [status, setStatus] = useState('');
  const [events, setEvents] = useState([]);
  const [stats, setStats] = useState({ progress: 0, fps: 0, people_count: 0, inside_count: 0, intrusion_count: 0 });
  const [busy, setBusy] = useState(false);
  const backendStatus = useBackendHealth();
  const socketsRef = useRef([]);
  const frameUrlRef = useRef('');
  const previewUrlRef = useRef('');
  const videoUrlRef = useRef('');

  const cleanupFrameUrl = () => {
    if (frameUrlRef.current) URL.revokeObjectURL(frameUrlRef.current);
    frameUrlRef.current = '';
  };

  const closeSockets = () => {
    socketsRef.current.forEach((socket) => socket.close());
    socketsRef.current = [];
  };

  useEffect(() => () => {
    closeSockets();
    cleanupFrameUrl();
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
  }, []);

  const handleFileChange = (nextFile) => {
    setFile(nextFile);
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    if (videoUrlRef.current) URL.revokeObjectURL(videoUrlRef.current);
    previewUrlRef.current = '';
    videoUrlRef.current = '';
    setPreviewUrl('');
    setVideoUrl('');
    cleanupFrameUrl();
    setLiveFrameUrl('');
    if (!nextFile) return;

    const url = URL.createObjectURL(nextFile);
    videoUrlRef.current = url;
    setVideoUrl(url);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.src = url;
    video.onloadeddata = () => {
      video.currentTime = Math.min(0.1, video.duration || 0);
    };
    video.onseeked = () => {
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 360;
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (!blob) return;
        if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
        const frameUrl = URL.createObjectURL(blob);
        previewUrlRef.current = frameUrl;
        setPreviewUrl(frameUrl);
      }, 'image/jpeg', 0.88);
    };
  };

  const connectJobSockets = (nextJobId) => {
    closeSockets();
    let attempts = 0;
    const connectStatus = () => {
      const socket = new WebSocket(`${WS_BASE}/ws/jobs/${nextJobId}`);
      socketsRef.current.push(socket);
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === 'progress') setStats((current) => ({ ...current, progress: message.data.progress ?? current.progress }));
        if (message.type === 'statistics') setStats((current) => ({ ...current, ...message.data }));
        if (message.type === 'intrusion') setEvents((current) => [message.data, ...current]);
        if (message.type === 'completed') setStatus(message.data.status || 'COMPLETED');
        if (message.type === 'error') setStatus('FAILED');
      };
      socket.onclose = () => {
        attempts += 1;
        if (attempts <= 3) window.setTimeout(connectStatus, attempts * 1000);
      };
    };
    connectStatus();

    const frameSocket = new WebSocket(`${WS_BASE}/ws/jobs/${nextJobId}/frames`);
    frameSocket.binaryType = 'blob';
    frameSocket.onmessage = (event) => {
      cleanupFrameUrl();
      const url = URL.createObjectURL(event.data);
      frameUrlRef.current = url;
      setLiveFrameUrl(url);
    };
    socketsRef.current.push(frameSocket);
  };

  const start = async () => {
    if (!file || points.length < 3) return;
    setBusy(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const uploadResponse = await fetch(`${API_BASE}/api/videos`, { method: 'POST', body: form });
      if (!uploadResponse.ok) throw new Error(await uploadResponse.text());
      const { video_id } = await uploadResponse.json();
      const jobResponse = await fetch(`${API_BASE}/api/jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id, confidence: 0.35, confirm_frames: 5, zone_points: points })
      });
      if (!jobResponse.ok) throw new Error(await jobResponse.text());
      const created = await jobResponse.json();
      setJobId(created.job_id);
      setStatus('QUEUED');
      connectJobSockets(created.job_id);
    } catch (error) {
      setStatus(`ERROR: ${error.message}`);
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    if (!jobId) return;
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}/stop`, { method: 'POST' });
    if (response.ok) {
      const snapshot = await response.json();
      setStatus(snapshot.status);
    }
  };

  const reset = () => {
    closeSockets();
    cleanupFrameUrl();
    setLiveFrameUrl('');
    setJobId('');
    setStatus('');
    setEvents([]);
    setStats({ progress: 0, fps: 0, people_count: 0, inside_count: 0, intrusion_count: 0 });
    setPoints([]);
  };

  return (
    <main className="app-shell">
      <header>
        <h1>Restricted Area Intrusion Detection</h1>
      </header>
      <div className="workspace">
        <VideoPanel
          file={file}
          previewUrl={previewUrl || videoUrl}
          liveFrameUrl={liveFrameUrl}
          points={points}
          onAddPoint={(point) => setPoints((current) => [...current, point])}
          onClearPoints={() => setPoints([])}
        />
        <aside>
          <ControlPanel
            onFileChange={handleFileChange}
            onStart={start}
            onStop={stop}
            onReset={reset}
            disabled={busy || backendStatus !== 'connected' || !file}
            status={status}
            backendStatus={backendStatus}
            pointsCount={points.length}
          />
          <StatisticsPanel stats={stats} />
        </aside>
      </div>
      <EventTable events={events} />
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);
