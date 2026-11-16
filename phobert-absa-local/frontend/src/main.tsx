import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';

type Prediction = {
  aspect: string;
  sentiment: string;
  confidence: number;
  scores: Record<string, number>;
};

const defaultAspects = [
  'GENERAL', 'PERFORMANCE', 'BATTERY', 'FEATURES', 'CAMERA',
  'PRICE', 'SER&ACC', 'DESIGN', 'SCREEN', 'STORAGE'
];

const names: Record<string, string> = {
  GENERAL: 'Tổng quan',
  PERFORMANCE: 'Hiệu năng',
  BATTERY: 'Pin',
  FEATURES: 'Tính năng',
  CAMERA: 'Camera',
  PRICE: 'Giá',
  'SER&ACC': 'Dịch vụ & phụ kiện',
  DESIGN: 'Thiết kế',
  SCREEN: 'Màn hình',
  STORAGE: 'Bộ nhớ'
};

const sentiments: Record<string, string> = {
  Positive: 'Tích cực',
  Negative: 'Tiêu cực',
  Neutral: 'Trung tính'
};

function App() {
  const [comment, setComment] = useState('Camera chụp ảnh rất đẹp nhưng pin nhanh hết.');
  const [aspectList, setAspectList] = useState<string[]>(defaultAspects);
  const [selected, setSelected] = useState<string[]>(['CAMERA', 'BATTERY']);
  const [results, setResults] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [serverStatus, setServerStatus] = useState<string | null>(null);

  useEffect(() => {
    // Kiểm tra health và lấy aspects từ backend khi khởi động
    fetch('http://127.0.0.1:8000/api/health')
      .then(res => res.json())
      .then(data => {
        if (data.status === 'ok') {
          setServerStatus(data.model_ready ? 'Sẵn sàng' : 'Online (Chờ nạp model)');
        }
      })
      .catch(() => setServerStatus('Chưa kết nối'));

    fetch('http://127.0.0.1:8000/api/aspects')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data.aspects) && data.aspects.length > 0) {
          setAspectList(data.aspects);
        }
      })
      .catch(() => {
        // Sử dụng defaultAspects nếu backend chưa mở
      });
  }, []);

  function toggleAspect(aspect: string) {
    setSelected(prev =>
      prev.includes(aspect) ? prev.filter(x => x !== aspect) : [...prev, aspect]
    );
  }

  function selectAll() {
    setSelected([...aspectList]);
  }

  function deselectAll() {
    setSelected([]);
  }

  async function analyze() {
    const trimmed = comment.trim();
    if (!trimmed) {
      setError('Vui lòng nhập nội dung đánh giá smartphone.');
      return;
    }
    if (selected.length === 0) {
      setError('Vui lòng chọn ít nhất một khía cạnh cần phân tích.');
      return;
    }

    setLoading(true);
    setError('');
    setResults([]);

    try {
      const res = await fetch('http://127.0.0.1:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ comment: trimmed, aspects: selected })
      });

      const data = await res.json();
      if (!res.ok) {
        let msg = 'Lỗi yêu cầu API';
        if (typeof data.detail === 'string') {
          msg = data.detail;
        } else if (Array.isArray(data.detail)) {
          msg = data.detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ');
        }
        throw new Error(msg);
      }

      setResults(data.results || []);
      setServerStatus('Sẵn sàng');
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Không thể kết nối đến máy chủ backend (http://127.0.0.1:8000).');
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="wrap">
      <header>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="eyebrow">LOCAL AI • PHOBERT-BASE-V2</div>
          {serverStatus && (
            <span style={{ fontSize: 12, padding: '4px 10px', borderRadius: 20, background: serverStatus === 'Sẵn sàng' ? '#184638' : '#332615', color: serverStatus === 'Sẵn sàng' ? '#85ebbe' : '#f1c876' }}>
              Backend: {serverStatus}
            </span>
          )}
        </div>
        <h1>Phân tích cảm xúc đánh giá smartphone</h1>
        <p>Phân tích cảm xúc theo khía cạnh (ABSA) bằng mô hình PhoBERT đã fine-tune, chạy hoàn toàn cục bộ trên máy tính.</p>
      </header>

      <section className="panel">
        <label htmlFor="comment">Nội dung đánh giá</label>
        <textarea
          id="comment"
          rows={5}
          value={comment}
          maxLength={10000}
          onChange={e => setComment(e.target.value)}
          placeholder="Nhập đánh giá smartphone..."
        />

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <label style={{ margin: 0 }}>Chọn khía cạnh cần phân tích ({selected.length}/{aspectList.length})</label>
          <div style={{ display: 'flex', gap: 8 }}>
            <button type="button" onClick={selectAll} style={{ background: 'transparent', border: 'none', color: '#8bb0ff', fontSize: 13, cursor: 'pointer', padding: 0 }}>Chọn tất cả</button>
            <span style={{ color: '#556' }}>•</span>
            <button type="button" onClick={deselectAll} style={{ background: 'transparent', border: 'none', color: '#8bb0ff', fontSize: 13, cursor: 'pointer', padding: 0 }}>Bỏ chọn</button>
          </div>
        </div>

        <div className="chips">
          {aspectList.map(a => (
            <button
              key={a}
              type="button"
              className={selected.includes(a) ? 'chip active' : 'chip'}
              aria-pressed={selected.includes(a)}
              onClick={() => toggleAspect(a)}
            >
              {names[a] || a}
            </button>
          ))}
        </div>

        <button className="run" disabled={loading} onClick={analyze}>
          {loading ? 'Đang phân tích...' : 'Phân tích cảm xúc'}
        </button>

        {error && <p className="error" role="alert" style={{ marginTop: 14 }}>{error}</p>}
      </section>

      {results.length > 0 && (
        <section className="panel">
          <h2>Kết quả phân tích</h2>
          <div className="results">
            {results.map(r => (
              <article key={r.aspect}>
                <div>
                  <strong>{names[r.aspect] || r.aspect}</strong>
                  <span className={'mood ' + r.sentiment}>
                    {sentiments[r.sentiment] || r.sentiment}
                  </span>
                </div>
                <p>Độ tin cậy: {(r.confidence * 100).toFixed(1)}%</p>
                <div className="track">
                  <div style={{ width: `${r.confidence * 100}%` }} />
                </div>
                {r.scores && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10, fontSize: 11, color: '#8ca1c2' }}>
                    <span>Tích cực: {(r.scores.Positive * 100).toFixed(1)}%</span>
                    <span>Trung tính: {(r.scores.Neutral * 100).toFixed(1)}%</span>
                    <span>Tiêu cực: {(r.scores.Negative * 100).toFixed(1)}%</span>
                  </div>
                )}
              </article>
            ))}
          </div>
          <small>
            Lưu ý: Mô hình chỉ dự đoán cảm xúc cho các khía cạnh bạn đã chọn, không xác định khía cạnh có thực sự được đề cập trong câu hay không.
          </small>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
