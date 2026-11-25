import { useEffect, useState } from 'react'
import axios from 'axios'
import './App.css'

const API = 'http://localhost:8000/api/v1'

const unitLabel = (unit) => (unit === 'degC' ? '°C' : 'DN (görece)')

function Histogram({ dist }) {
  if (!dist?.available) return null
  const max = Math.max(...dist.counts, 1)
  const lo = dist.bin_edges[0]
  const hi = dist.bin_edges[dist.bin_edges.length - 1]
  return (
    <div className="histogram" role="img" aria-label="Bölge içi değer dağılımı histogramı">
      <div className="hist-bars">
        {dist.counts.map((c, i) => (
          <div key={i} className="hist-bar" style={{ height: `${(c / max) * 100}%` }}
            title={`${dist.bin_edges[i]} – ${dist.bin_edges[i + 1]}: ${c} piksel`} />
        ))}
      </div>
      <div className="hist-axis">
        <span>{lo}</span><span>{unitLabel(dist.unit)}</span><span>{hi}</span>
      </div>
      <small>Yüzdelikler: %10 = {dist.p10}, %50 = {dist.p50}, %90 = {dist.p90}</small>
    </div>
  )
}

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])
  const [historyError, setHistoryError] = useState(null)
  const [selectedRecord, setSelectedRecord] = useState(null)
  const isVideo = Boolean(file && /\.(mp4|avi|mov|mkv)$/i.test(file.name))

  const loadHistory = async () => {
    try {
      const response = await axios.get(`${API}/analyses?limit=50`)
      setHistory(response.data.items || [])
      setHistoryError(null)
    } catch (err) {
      setHistoryError(err.response?.data?.detail || 'Analiz geçmişi yüklenemedi')
    }
  }

  useEffect(() => {
    loadHistory()
  }, [])

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview)
  }, [preview])

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0] || null
    setFile(selected)
    setResult(null)
    setError(null)
    setPreview(selected && selected.type.startsWith('image/')
      ? URL.createObjectURL(selected) : null)
  }

  const handleAnalyze = async () => {
    if (!file) {
      setError('Lütfen bir termal görüntü veya video seçin')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const endpoint = isVideo ? 'analyze-video' : 'analyze'
      const response = await axios.post(`${API}/${endpoint}`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: isVideo ? 180000 : 60000,
      })
      setResult(response.data)
      await loadHistory()
    } catch (err) {
      setError(err.response?.data?.detail || 'Analiz sırasında hata oluştu')
    } finally {
      setLoading(false)
    }
  }

  const openRecord = async (id) => {
    try {
      const response = await axios.get(`${API}/analyses/${id}`)
      setSelectedRecord(response.data)
    } catch (err) {
      setHistoryError(err.response?.data?.detail || 'Analiz kaydı açılamadı')
    }
  }

  const candidates = result?.candidates || []
  const tracks = result?.tracks || []

  return (
    <div className="container">
      <h1>Termal Kamera Analiz Sistemi</h1>
      <p className="subtitle">Kayıp Personel Aday Bölgesi Tespiti ve Termal Analiz</p>
      <div className="upload-box">
        <input type="file" accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,.mp4,.avi,.mov,.mkv,image/*,video/*" onChange={handleFileChange} />
        {preview && <div className="preview"><img src={preview} alt="Yüklenen termal görüntü önizlemesi" /></div>}
        <button onClick={handleAnalyze} disabled={!file || loading} className="upload-btn">
          {loading ? 'Analiz ediliyor...' : isVideo ? 'Videoyu Analiz Et' : 'Görüntüyü Analiz Et'}
        </button>
      </div>
      {error && <div className="error"><strong>Hata:</strong> {error}</div>}

      {result && (
        <div className="success">
          <h3>{result.video ? 'Video Analiz Sonucu' : 'Görüntü Analiz Sonucu'}</h3>
          <p><strong>Analiz kayıt numarası:</strong> #{result.analysis_id}</p>
          <p><strong>Dosya:</strong> {result.original_filename}</p>
          <a className="pdf-link" href={`${API}/analyses/${result.analysis_id}/report.pdf`}>PDF raporunu indir</a>
          {!result.video && <>
            <p><strong>Boyut:</strong> {result.image_shape?.join(' × ')}</p>
            <p><strong>Yöntem:</strong> {result.method}</p>
            <p className="person-count"><strong>Aday bölge sayısı:</strong> {result.candidate_count}</p>
            <p className="warning">Bunlar doğrulanmış insan tespiti değil, klasik görüntü işlemeyle bulunan aday bölgelerdir.</p>
            {result.annotated_image && <div className="annotated"><img src={result.annotated_image} alt="Aday bölgelerin işaretlendiği sonuç görüntüsü" /></div>}
            {candidates.map((p, i) => <p key={i}>Aday {i + 1}: x={p.x}, y={p.y}, w={p.w}, h={p.h}; skor={p.score}</p>)}
            <h4>Sıcaklık / Piksel Yoğunluğu Analizi</h4>
            <p className="warning">{result.temperature_warning}</p>
            {result.heatmap_image && (
              <div className="heatmap">
                <h5>Isı haritası (görece yoğunluk)</h5>
                <div className="heatmap-row">
                  <img src={result.heatmap_image} alt="Renk haritasıyla gösterilen görece piksel yoğunluğu" />
                  <div className="heatmap-legend"><span>yüksek</span><span>düşük</span></div>
                </div>
                <small>{result.heatmap_note}</small>
              </div>
            )}
            {result.temperature_analysis?.map((item, i) => (
              <div className="candidates" key={i}>
                <h5>Aday {i + 1}</h5>
                <p>Ortalama {item.mean?.toFixed(2)} {unitLabel(item.unit)}, min {item.min?.toFixed(2)}, max {item.max?.toFixed(2)}, std {item.std?.toFixed(2)}</p>
                {item.background_comparison?.available && (
                  <p><strong>Çevre karşılaştırması:</strong> aday {item.background_comparison.candidate_mean}, çevre {item.background_comparison.background_mean}, fark {item.background_comparison.difference} {unitLabel(item.background_comparison.unit)}
                    {item.background_comparison.note ? ` — ${item.background_comparison.note}` : ''}</p>
                )}
                <Histogram dist={item.distribution} />
              </div>
            ))}
            <h4>Yaşam Belirtisi</h4>
            <p className="finding"><strong>{result.vital_signs?.finding}</strong></p>
            <p>{result.vital_signs?.message}</p>
          </>}
          {result.video && <>
            <p>Süre: {result.video.duration_s} s | Kaynak FPS: {result.video.source_fps} | Analiz FPS: {result.video.analysis_fps}</p>
            <p>İşlenen kare: {result.video.sampled_frames} / {result.video.frame_count}</p>
            <p>Aday takip sayısı: {result.track_count}</p>
            <p className="warning">{result.interpretation_warning}</p>
            {result.preview_frames?.length > 0 && (
              <div className="video-previews">
                <h4>İşaretli kare önizlemeleri</h4>
                <div className="preview-grid">
                  {result.preview_frames.map((f, i) => (
                    <figure key={i}>
                      <img src={f.image} alt={`Videonun ${f.time_s}. saniyesindeki işaretli kare`} />
                      <figcaption>{f.time_s} sn</figcaption>
                    </figure>
                  ))}
                </div>
              </div>
            )}
            {tracks.map((track) => (
              <div className="candidates" key={track.track_id}>
                <h4>Aday takip #{track.track_id}</h4>
                <p>Gözlem: {track.candidate_observations} | Kapsama: %{Math.round(track.coverage * 100)}</p>
                <p>Sinyal: {track.signal_description} ({track.signal_unit})</p>
                <p className="finding"><strong>{track.vital_signs?.finding}</strong></p>
                <p>{track.vital_signs?.message}</p>
                {track.vital_signs?.details && <p>Dominant frekans: {track.vital_signs.details.dominant_frequency_hz} Hz | Bant içi oran: {track.vital_signs.details.respiratory_band_rate_per_min} / dakika | Tepe / arka plan: {track.vital_signs.details.peak_to_trend_db} dB (eşik {track.vital_signs.details.threshold_db} dB)</p>}
              </div>
            ))}
          </>}
        </div>
      )}

      <section className="history-panel">
        <div className="history-heading">
          <h2>Analiz Geçmişi</h2>
          <button className="refresh-btn" onClick={loadHistory}>Yenile</button>
        </div>
        {historyError && <p className="warning">{historyError}</p>}
        {history.length === 0 ? <p>Henüz kayıtlı analiz bulunmuyor.</p> : (
          <div className="history-list">
            {history.map((item) => (
              <div className="history-item" key={item.id}>
                <div>
                  <strong>#{item.id} · {item.original_filename}</strong>
                  <span>{item.analysis_type.toUpperCase()} · {item.candidate_count ?? 0} aday · {item.created_at ? new Date(item.created_at).toLocaleString() : ''}</span>
                </div>
                <div className="history-actions">
                  <button className="refresh-btn" onClick={() => openRecord(item.id)}>Detay</button>
                  <a className="pdf-link" href={`${API}/analyses/${item.id}/report.pdf`}>PDF</a>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {selectedRecord && (
        <section className="record-detail">
          <div className="history-heading">
            <h3>Analiz #{selectedRecord.id}</h3>
            <button className="refresh-btn" onClick={() => setSelectedRecord(null)}>Kapat</button>
          </div>
          <p><strong>Dosya:</strong> {selectedRecord.original_filename}</p>
          <p><strong>Tür:</strong> {selectedRecord.analysis_type}</p>
          <p><strong>Tarih:</strong> {selectedRecord.created_at}</p>
          <pre>{JSON.stringify(selectedRecord.result_data, null, 2)}</pre>
          <a className="pdf-link" href={`${API}/analyses/${selectedRecord.id}/report.pdf`}>Bu analizin PDF raporunu indir</a>
        </section>
      )}

      <div className="info">
        <p>Desteklenen görüntüler: JPG, PNG, TIFF, BMP. Videolar: MP4, AVI, MOV, MKV.</p>
        <p>Analiz kayıtları PostgreSQL veritabanında saklanır.</p>
        <p>Backend: <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API Docs</a></p>
      </div>
    </div>
  )
}
export default App
