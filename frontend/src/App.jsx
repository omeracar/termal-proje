import { useEffect, useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

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
      setError('Lütfen bir termal görüntü dosyası seçin')
      return
    }
    setLoading(true)
    setError(null)
    setResult(null)
    const formData = new FormData()
    formData.append('file', file)
    try {
      const response = await axios.post('http://localhost:8000/api/v1/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Analiz sırasında hata oluştu')
    } finally {
      setLoading(false)
    }
  }

  const candidates = result?.candidates || []

  return (
    <div className="container">
      <h1>Termal Kamera Analiz Sistemi</h1>
      <p className="subtitle">Kayıp Personel Aday Bölgesi Tespiti ve Termal Analiz</p>

      <div className="upload-box">
        <input type="file" accept=".jpg,.jpeg,.png,.tif,.tiff,.bmp,image/*" onChange={handleFileChange} />
        {preview && <div className="preview"><img src={preview} alt="Yüklenen termal görüntü önizlemesi" /></div>}
        <button onClick={handleAnalyze} disabled={!file || loading} className="upload-btn">
          {loading ? 'Analiz ediliyor...' : 'Analiz Et'}
        </button>
      </div>

      {error && <div className="error"><strong>Hata:</strong> {error}</div>}

      {result && (
        <div className="success">
          <h3>Analiz Sonucu</h3>
          <p><strong>Dosya:</strong> {result.original_filename}</p>
          <p><strong>Görüntü boyutu:</strong> {result.image_shape?.join(' × ')}</p>
          <p><strong>Yöntem:</strong> {result.method}</p>
          <p className="person-count"><strong>Aday bölge sayısı:</strong> {result.candidate_count}</p>
          <p className="warning">
            Bu kutular klasik görüntü işleme ile bulunan termal/geometrik aday bölgelerdir;
            doğrulanmış insan tespiti veya kişi sayısı değildir.
          </p>
          {result.annotated_image && (
            <div className="annotated"><img src={result.annotated_image} alt="Aday bölgelerin işaretlendiği sonuç görüntüsü" /></div>
          )}
          {candidates.length > 0 ? (
            <div className="candidates">
              <h4>Aday Bölgeler</h4>
              <ul>{candidates.map((p, i) => (
                <li key={i}>
                  Aday {i + 1}: x={p.x}, y={p.y}, w={p.w}, h={p.h};
                  geometrik sıralama skoru={p.score}
                </li>
              ))}</ul>
            </div>
          ) : <p>Belirlenen eşik ve geometri koşullarına uyan aday bölge bulunamadı.</p>}
          <h4>Sıcaklık Analizi</h4>
          <p>{result.temperature_warning}</p>
          {result.temperature_analysis?.map((item, i) => (
            <p key={i}>Aday {i + 1}: {item.measurement_type} — ortalama {item.mean?.toFixed(2)} {item.unit},
              min {item.min?.toFixed(2)}, max {item.max?.toFixed(2)}</p>
          ))}
          <h4>Yaşam Belirtisi / Solunum Sinyali</h4>
          <p>{result.vital_signs?.message}</p>
          <p className="warning">
            Bu prototip tıbbi cihaz değildir. Sonuçlar yaşam veya ölüm kararı vermek için kullanılamaz.
          </p>
        </div>
      )}

      <div className="info">
        <p>Desteklenen analiz formatları: JPG, PNG, TIFF, BMP</p>
        <p>Video analizi henüz arayüze/API'ye entegre edilmemiştir.</p>
        <p>Backend: <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API Docs</a></p>
      </div>
    </div>
  )
}
export default App
