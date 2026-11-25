# Termal Kamera ile Kayıp Personel Tespiti – Karar Destek Prototipi

Termal görüntülerde **insan olabilecek bölgeleri** klasik görüntü işleme yöntemleriyle bulan, sonuçları PostgreSQL'e kaydeden ve PDF rapor üreten bir araştırma prototipidir.

> **Yapay zekâ, makine öğrenmesi, hazır model ya da model indirme kullanılmaz.** Yalnızca eşikleme, morfoloji, kontur/geometri ve sinyal işleme yöntemleri vardır.

## Uygulananlar

- Klasik OpenCV ile aday bölge çıkarımı: gürültü azaltma, yerel kontrast haritası, çok seviyeli adaptif eşikleme, morfoloji, kontur/geometri filtreleri, yan yana duran kişilerin ayrılması, parçalanmış kişilerin birleştirilmesi ve arayüz grafiklerinin (renk skalası, yazı) elenmesi.
- Aday bölgeler açıkça "doğrulanmış kişi" olarak etiketlenmez.
- Ham piksel yoğunluğu (DN) ile kalibre edilmiş radyometrik sıcaklık (°C) birbirinden ayrı tutulur.
- Görüntü ve video analizi uç noktaları.
- Her aday için bölge istatistiği (ortalama, min, max, std), **çevre karşılaştırması** (aday ile çevresindeki halka) ve **değer dağılımı histogramı**. Kalibre edilmiş °C haritası yoksa tüm değerler "DN (görece)" etiketiyle gösterilir ve fiziksel sıcaklık olarak yorumlanmamalıdır.
- **Isı haritası** (renk skalalı, görece yoğunluk) ve videoda iz kutularıyla **işaretli önizleme kareleri** (ilk, orta, son).
- Yaşam belirtisi sonucu şartnamedeki üç ifadeden biriyle verilir: "Yaşam belirtisiyle uyumlu bulgu tespit edildi.", "Yaşam belirtisi belirlenemedi.", "Analiz için yeterli veri bulunmuyor." (`vital_signs.finding`).
- PDF rapor, varsa `data/eval_metrics.json` içindeki dedektör doğrulama ölçümlerini de içerir.
- Sezgisel video ilgi bölgesi (ROI) takibi ve keşif amaçlı Welch PSD sinyal analizi.
- Görüntü ve video sonuçlarının PostgreSQL'e kaydı.
- Analiz geçmişi ve ayrıntı uç noktaları.
- Her kayıtlı analiz için indirilebilir PDF rapor.
- Yükleme, geçmiş, kayıt ayrıntıları ve PDF indirme için React arayüzü.

## PostgreSQL kurulumu

Depoda `docker-compose.yml` içinde yerel bir PostgreSQL servisi vardır.

1. Depo kök klasöründen PostgreSQL'i başlatın:
```bash
docker compose up -d postgres
```

2. Arka uç ortam dosyasını oluşturun:
```bash
cd backend
cp .env.example .env
```

3. Gereksinimleri kurun ve API'yi başlatın:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Örnek bağlantı adresi yerel Docker PostgreSQL servisini hedefler:
`postgresql+psycopg2://thermal_user:thermal_password@localhost:5432/thermal_camera`

API açılırken SQLAlchemy gerekli tabloları yoksa oluşturur. Bu prototipte şema oluşturma `create_all` ile yapılır; üretim ortamına geçmeden önce sürümlü bir Alembic göç (migration) düzeni eklenmelidir.

PostgreSQL olmadan hızlı yerel geliştirme için `backend/.env` dosyasını silin ya da yeniden adlandırın; uygulama yerel bir SQLite dosyasına geçer.

## Analiz geçmişi ve PDF uç noktaları

- `POST /api/v1/analyze` — görüntüyü analiz eder ve sonucu kaydeder.
- `POST /api/v1/analyze-video` — videoyu analiz eder ve sonucu kaydeder.
- `GET /api/v1/analyses?limit=50&offset=0` — kayıtlı analizleri listeler.
- `GET /api/v1/analyses/{id}` — tek bir analizi ve saklanan sonuç verisini getirir.
- `GET /api/v1/analyses/{id}/report.pdf` — PDF raporu indirir.

PDF raporları analiz üst verisini, özet ölçütleri, teknik sonuç verisini ve yorum/güvenlik sınırlamalarını içerir. Sunucuda DejaVu Sans, Liberation Sans ya da Arial gibi Unicode destekli bir yazı tipi varsa Türkçe karakterler korunur; yoksa ASCII'ye çevrilir.

## Arayüz

`frontend` klasöründen:
```bash
npm install
npm run dev
```

Arayüz analiz numarasını, yenilenebilir analiz geçmişini, kayıt ayrıntılarını ve PDF indirme bağlantılarını gösterir.

## Dedektör değerlendirmesi

Gerçek Precision/Recall/F1 değerleri için elle etiketlenmiş gerçek veri gerekir. Görüntüleri depodaki fareyle çalışan araçla etiketleyin:

1. `tools/label_boxes.html` dosyasını tarayıcıda açın (çift tıklama yeterli; sunucu ya da internet gerekmez).
2. **Klasör seç** düğmesiyle `data/samples` klasörünü seçin (tek dosyalar için **Görüntü seç**).
3. Görünen her kişinin etrafına sıkı bir kutu sürükleyerek çizin. `E` tuşu görüntüyü "insan yok" olarak işaretler, `Z` geri alır, `Delete` seçili kutuyu siler, ok tuşları görüntüler arasında geçer. İlerleme tarayıcıda otomatik kaydedilir.
4. **Dışa aktar** düğmesine basın ve inen `ground_truth.json` dosyasını `data/samples/` içine koyun. Yalnızca gözden geçirilmiş görüntüler dışa aktarılır.
5. Değerlendirmeyi çalıştırın (`--vis` ile sonuç resimleri de üretilir: yeşil = doğru, kırmızı = yanlış tespit, sarı = kaçırılan kişi):

```bash
cd backend
python scripts/evaluate_detector.py --images ../data/samples --truth ../data/samples/ground_truth.json --vis ../data/eval_vis --out ../data/eval_metrics.json
```

Kutular görüntü pikseli cinsinden `[x, y, genişlik, yükseklik]` biçimindedir; IoU ≥ 0,5 olan tespit doğru sayılır (bire bir açgözlü eşleştirme). Etiket dosyası **Etiket yükle (JSON)** ile araca geri yüklenip etiketlemeye devam edilebilir.

### Ölçülen başarım (10 elle etiketli görüntü, IoU ≥ 0,5)

| Ölçüt | Değer |
|---|---|
| Etiketli görüntü / kişi | 10 görüntü, 55 kişi |
| Doğru tespit (TP) | 14 |
| Yanlış tespit (FP) | 31 |
| Tespit edilemeyen kişi (FN) | 41 |
| **Precision** | **0,311** |
| **Recall** | **0,255** |
| **F1** | **0,280** |
| Görüntü başına süre | 0,107 s |

Ayrıntılı görüntü bazlı sonuçlar `data/eval_metrics.json` dosyasındadır. 10.jpeg tek başına 25 kişilik uzak, yoğun bir kalabalıktır; o görüntü çıkarılınca Precision ≈ 0,35, Recall ≈ 0,43, F1 ≈ 0,39 olur. Örneklem küçük ve tek kaynaklıdır; bu değerler genelleştirilemez. Dedektörün eşikleri bu etiketlere göre **ayarlanmamıştır**. Sıcaklık ölçüm hatası ve solunum tespit başarısı için referans veri bulunmadığından ölçülememiştir.

Önceki gayriresmî kontrol (etiketsiz), 11 gerçek termal görüntü (kişiler gözle sayıldı, etiketli doğru veri DEĞİL; dolayısıyla Precision/Recall değeri değildir): dedektör 3 görüntüde sayıyı tam bildi, 3 görüntüde bir kişi farkla bildi, kalan 5 görüntüde üç veya daha fazla fark verdi (hayvanlar, yakın plan özneler, uzak ya da yoğun kalabalıklar, 7 kişi için 12 aday çıkan kalabalık bir sokak sahnesi). Ortalama süre işlemcide görüntü başına yaklaşık 0,12 saniyedir.

## Veri seti belgelendirmesi

Şartname, veri setlerinin kaynaklarının, erişim bağlantılarının ve teknik özelliklerinin belgelenmesini zorunlu tutar. **Bu belge henüz tamamlanmamıştır.**

`data/samples` içindeki 10 görüntü gerçek termal kamera görüntüleridir, ancak kaynakları, lisansları ve çekim bilgileri henüz kayıt altına alınmamıştır. Aşağıdaki tablo doldurulmalıdır:

| Dosya | Veri seti / kaynak | Erişim bağlantısı | Lisans | Radyometrik mi? | Video var mı? |
|---|---|---|---|---|---|
| 1.jpg … 10.jpeg | _belirtilecek_ | _belirtilecek_ | _belirtilecek_ | _belirtilecek_ | _belirtilecek_ |

Şartnamede önerilen kaynaklar (FLIR Thermal Dataset, KAIST Multispectral Pedestrian Dataset, LLVIP, Zenodo, IEEE DataPort, akademik veri setleri) kullanılmadan önce şu açılardan incelenmelidir: radyometrik sıcaklık içeriyor mu, termal video içeriyor mu, solunum doğrulaması için etiket var mı. Bunlar bu depoda henüz doğrulanmamıştır.

## Şartnameye göre eksikler (güncel)

- **Gerçek sıcaklık (°C):** Kameraya özgü radyometrik veri çözücü yoktur. Sistem fiilen yalnızca DN (görece) değerleri üretir; çevre karşılaştırması ve histogram da DN cinsindendir.
- **Yaşam belirtisi:** Etiketli gerçek termal video ile hiç doğrulanmamıştır. Bulgu yalnızca keşif amaçlıdır.
- **Başarım:** Precision 0,31 / Recall 0,26 ölçülmüştür; dedektör iyileştirmesi yapılmamıştır. Farklı çevre koşullarında karşılaştırmalı inceleme yapılmamıştır.
- **Veri seti belgelendirmesi:** Yukarıdaki tablo boştur.

## Önemli teknik sınırlamalar

1. Aday bölgeler doğrulanmış kişi değildir. Eşikleme ve geometri, insanı sıcak nesnelerden, hayvanlardan, yansımalardan ya da araziden güvenilir biçimde ayıramaz.
2. Video ROI eşleştirmesi basit bir en yakın merkez sezgisidir; kamera hareketi, örtülme ve kesişen kişiler iz kimliğini bozabilir.
3. Mevcut zamansal sinyal, kalibre edilmiş sıcaklık değil, DN cinsinden gri seviye piksel yoğunluğudur.
4. Solunum bandındaki periyodiklik keşif amaçlıdır ve gerçek veriyle doğrulanmamıştır. Algılayıcı, tepe gücünü sinyalin kendi 1/f arka plan eğrisiyle karşılaştırır ve en az 3 tam döngü ister. Yalnızca simülasyonla sınanmıştır: hiç solunum yokken (beyaz/pembe gürültü, rastgele sürüklenme) yanlış olumlu oranı 20 sn'lik sinyalde yaklaşık %2–4, 60 sn'lik sinyalde yaklaşık %0 çıkmıştır; gürültü genliğine eşit bir periyodik bileşen yaklaşık %95+ yakalanır. Gerçek kamera gürültüsü (otomatik kazanç, hareket, hava akımı) bunlardan farklıdır, bu oranlar gerçek başarım değildir. Olumlu bileşen nefes almanın ya da yaşamın kanıtı değildir; olumsuz sonuç da ölümün kanıtı değildir.
5. API'ye kameraya özgü bir radyometrik çözücü/kalibrasyon adaptörü bağlı değildir.
6. Yalnızca 10 etiketli görüntüyle bir ilk ölçüm yapılmıştır (Precision 0,31, Recall 0,26); sıcaklık hatası ve solunum algılama başarımı bilinmemektedir ("Dedektör değerlendirmesi" bölümüne bakın). Hayvanlar insanlardan ayrılamaz, yakın plan özneler parçalara bölünebilir, küçük öznelerden oluşan yoğun kalabalıklar eksik sayılır.
7. PostgreSQL analiz üst verisini ve sonuç JSON'unu saklar. Yüklenen kaynak medya dosyalarının kendisi veritabanında ya da nesne depolamada arşivlenmez.

## Güvenlik

Bu bir araştırma prototipidir; tıbbi cihaz ya da doğrulanmış bir arama-kurtarma yaşam algılayıcısı değildir. Çıktısı, kurtarma, triyaj ya da yaşam/ölüm kararlarında asla tek dayanak olarak kullanılmamalıdır. İnsan denetimi ve bağımsız kurtarma prosedürleri vazgeçilmezdir.

API belgeleri: `http://localhost:8000/docs`
