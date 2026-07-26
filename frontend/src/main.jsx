import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Play, RotateCcw, Square, Upload } from 'lucide-react';
import './styles.css';

const API_BASE = 'http://127.0.0.1:8000';
const WS_BASE = 'ws://127.0.0.1:8000';

function toApiUrl(path) {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}

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
      if (points.length > 0 && !liveFrameUrl) {
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
        {/* title="Clear polygon" -> title="Xóa vùng đa giác" */}
        <button type="button" className="icon-button" onClick={onClearPoints} title="Xóa vùng đa giác">
          <RotateCcw size={18} />
        </button>
      </div>
      <div className="video-stage">
        {(liveFrameUrl || previewUrl) ? (
          <div className="video-wrapper">
            <img ref={imageRef} src={liveFrameUrl || previewUrl} alt={file?.name || 'Xem trước video'} />
            <canvas ref={canvasRef} onClick={handleCanvasClick} />
          </div>
        ) : (
          /* 'Select a video' -> 'Vui lòng chọn một video' */
          <div className="empty-stage">Vui lòng chọn một video</div>
        )}
      </div>
    </section>
  );
}

function ControlPanel({ onFileChange, onStart, onStop, onReset, disabled, status, backendStatus, pointsCount, fastMode, onFastModeChange, confirmFrames, onConfirmFramesChange }) {
  return (
    <section className="control-panel">
      {/* <h2>Controls</h2> -> <h2>Bảng điều khiển</h2> */}
      <h2>Bảng điều khiển</h2>
      <label className="file-picker">
        <Upload size={18} />
        {/* <span>Choose video</span> -> <span>Chọn video</span> */}
        <span>Chọn video</span>
        <input type="file" accept="video/*" onChange={(event) => onFileChange(event.target.files?.[0] || null)} />
      </label>
      <div className="button-row">
        <button type="button" onClick={onStart} disabled={disabled || pointsCount < 3}>
          {/* Start -> Bắt đầu */}
          <Play size={16} /> Bắt đầu
        </button>
        <button type="button" onClick={onStop} disabled={!status || status === 'COMPLETED' || status === 'STOPPED'}>
          {/* Stop -> Dừng */}
          <Square size={16} /> Dừng
        </button>
        <button type="button" onClick={onReset}>
          {/* Reset -> Đặt lại */}
          <RotateCcw size={16} /> Đặt lại
        </button>
      </div>
      <label className="toggle-row">
        <input type="checkbox" checked={fastMode} onChange={(event) => onFastModeChange(event.target.checked)} />
        <span>Chế độ nhanh</span>
      </label>

      <div className="slider-row">
        <span>Số frame xác nhận: <strong>{confirmFrames}</strong></span>
        <input type="range" min="1" max="60" step="1" value={confirmFrames} onChange={(e) => onConfirmFramesChange(parseInt(e.target.value, 10))} />
      </div>
      <div className="status-strip">
        {/* <span>Backend</span> -> <span>Hệ thống nền</span> */}
        <span>Hệ thống nền</span>
        {/* connected -> Đã kết nối, disconnected -> Mất kết nối, checking -> Đang kiểm tra */}
        <strong className={backendStatus === 'connected' ? 'good' : 'bad'}>
          {backendStatus === 'connected' ? 'Đã kết nối' : backendStatus === 'disconnected' ? 'Mất kết nối' : 'Đang kiểm tra'}
        </strong>
      </div>
      <div className="status-strip">
        {/* <span>Job</span> -> <span>Tiến trình</span> */}
        <span>Tiến trình</span>
        {/* idle -> Đang rảnh, RUNNING -> Đang chạy, COMPLETED -> Hoàn thành, STOPPED -> Đã dừng, FAILED -> Thất bại, PENDING -> Đang chờ */}
        <strong>
          {status === 'RUNNING' ? 'Đang chạy' : 
           status === 'COMPLETED' ? 'Hoàn thành' : 
           status === 'STOPPED' ? 'Đã dừng' : 
           status === 'FAILED' ? 'Thất bại' : 
           status === 'PENDING' ? 'Đang chờ' : 
           status.startsWith('ERROR:') ? status.replace('ERROR:', 'LỖI:') : 'Đang treo'}
        </strong>
      </div>
    </section>
  );
}

function StatisticsPanel({ stats }) {
  /* 'Progress'    -> 'Tiến độ'
    'FPS'         -> 'Khung hình/giây (FPS)'
    'People'      -> 'Số người'
    'Inside'      -> 'Trong vùng'
    'Intrusions'  -> 'Số vụ xâm nhập'
  */
  const items = [
    ['Tiến độ', `${Math.round((stats.progress || 0) * 100)}%`],
    ['Khung hình/giây (FPS)', stats.fps.toFixed(1)],
    ['Số người', stats.people_count],
    ['Trong vùng', stats.inside_count],
    ['Số vụ xâm nhập', stats.intrusion_count]
  ];
  return (
    <section className="statistics-panel">
      {/* <h2>Statistics</h2> -> <h2>Thống kê</h2> */}
      <h2>Thống kê</h2>
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
      {/* <h2>Events</h2> -> <h2>Danh sách sự kiện</h2> */}
      <h2>Danh sách sự kiện</h2>
      <table>
        <thead>
          <tr>
            {/* Frame -> Khung hình
              Track -> ID Theo vết
              Confidence -> Độ tin cậy
              Evidence -> Bằng chứng
            */}
            <th>Khung hình</th>
            <th>ID Theo vết</th>
            <th>Độ tin cậy</th>
            <th>Bằng chứng</th>
          </tr>
        </thead>
        <tbody>
          {events.map((event, index) => (
            <tr key={`${event.frame}-${event.track_id}-${index}`}>
              <td>{event.frame}</td>
              <td>{event.track_id}</td>
              <td>{Number(event.confidence || 0).toFixed(2)}</td>
              <td>
                {event.evidence_url ? (
                  /* alt={`Evidence frame ${event.frame}`} -> alt={`Bằng chứng khung hình ${event.frame}`} */
                  <img className="evidence-thumb" src={toApiUrl(event.evidence_url)} alt={`Bằng chứng khung hình ${event.frame}`} title={event.evidence_path} />
                ) : event.evidence_path ? (
                  /* saved -> Đã lưu */
                  <span title={event.evidence_path}>Đã lưu</span>
                ) : (
                  '-'
                )}
              </td>
            </tr>
          ))}
          {events.length === 0 && (
            <tr>
              {/* 'No intrusion events' -> 'Không có sự kiện xâm nhập nào' */}
              <td colSpan="4" className="empty-table">Không có sự kiện xâm nhập nào</td>
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
  const [fastMode, setFastMode] = useState(true);
  const [confirmFrames, setConfirmFrames] = useState(15);
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
      // Đợi 250ms để trình duyệt giải mã xong frame trên GPU trước khi vẽ lên canvas
      setTimeout(() => {
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
      }, 250);
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
        if (message.type === 'progress') {
          setStatus('RUNNING');
          setStats((current) => ({ ...current, progress: message.data.progress ?? current.progress }));
        }
        if (message.type === 'statistics') {
          setStatus('RUNNING');
          setStats((current) => ({ ...current, ...message.data }));
        }
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
        body: JSON.stringify({
          video_id,
          confidence: 0.15,
          confirm_frames: confirmFrames,
          process_width: fastMode ? 640 : null,
          frame_stride: fastMode ? 3 : 1,
          save_output_video: !fastMode,
          zone_points: points,
          filter_type: "auto"
        })
      });
      if (!jobResponse.ok) throw new Error(await jobResponse.text());
      const created = await jobResponse.json();
      setJobId(created.job_id);
      setStatus('PENDING');
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
        {/* <h1>Restricted Area Intrusion Detection</h1> -> <h1>Hệ thống Phát hiện Xâm nhập Vùng cấm</h1> */}
        <h1>Hệ thống Phát hiện Xâm nhập Vùng cấm</h1>
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
            fastMode={fastMode}
            onFastModeChange={setFastMode}
            confirmFrames={confirmFrames}
            onConfirmFramesChange={setConfirmFrames}
          />
          <StatisticsPanel stats={stats} />
        </aside>
      </div>
      <EventTable events={events} />
    </main>
  );
}

createRoot(document.getElementById('root')).render(<App />);