import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const selected = e.target.files[0]
    setFile(selected)
    setResult(null)
    setError(null)

    if (selected && selected.type.startsWith('image/')) {
      const url = URL.createObjectURL(selected)
      setPreview(url)
    } else {
      setPreview(null)
    }
  }

  const handleAnalyze = async () => {
    if (!file) {
      setError("Lütfen bir görüntü dosyası seçin")
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await axios.post("http://localhost:8000/api/v1/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || "Analiz sırasında hata oluştu")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>Termal Kamera Analiz Sistemi</h1>
      <p className="subtitle">Kayıp Personel Tespiti ve Yaşam Belirtisi Analizi</p>

      <div className="upload-box">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
        />

        {preview && (
          <div className="preview">
            <img src={preview} alt="Önizleme" />
          </div>
        )}

        <button
          onClick={handleAnalyze}
          disabled={!file || loading}
          className="upload-btn"
        >
          {loading ? "Analiz ediliyor..." : "Analiz Et"}
        </button>
      </div>

      {error && (
        <div className="error">
          <strong>Hata:</strong> {error}
        </div>
      )}

      {result && (
        <div className="success">
          <h3>Analiz Sonucu</h3>
          <p><strong>Dosya:</strong> {result.original_filename}</p>
          <p><strong>Görüntü boyutu:</strong> {result.image_shape?.join(" × ")}</p>
          <p className="person-count">
            <strong>Tespit edilen kişi sayısı:</strong> {result.person_count}
          </p>
          {result.method === "threshold-legacy" && (
            <p className="warning">
              Yapay zekâ modeli yüklenemedi, düşük doğruluklu yedek yöntem kullanıldı.
            </p>
          )}

          {result.annotated_image && (
            <div className="annotated">
              <img src={result.annotated_image} alt="Kişilerin işaretlendiği sonuç görüntüsü" />
            </div>
          )}

          {result.people && result.people.length > 0 ? (
            <div className="candidates">
              <h4>Tespit Edilen Kişiler:</h4>
              <ul>
                {result.people.map((p, i) => (
                  <li key={i}>
                    Kişi {i + 1}: x={p.x}, y={p.y}, w={p.w}, h={p.h}
                    {" "}(güven: %{Math.round(p.score * 100)})
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p>Görüntüde kişi bulunamadı.</p>
          )}
        </div>
      )}

      <div className="info">
        <p>Desteklenen formatlar: JPG, PNG, TIFF, BMP</p>
        <p>Backend: <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">API Docs</a></p>
      </div>
    </div>
  )
}

export default App
