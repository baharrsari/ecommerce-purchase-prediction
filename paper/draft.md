# Session Bazlı E-Ticaret Satın Alma Tahmini ve Kategori Sınıflandırması: İki Aşamalı Bir Makine Öğrenmesi Yaklaşımı

**Bahar Sarımehmetoğlu**
*Ankara Üniversitesi*
*Bilgisayar Mühendisliği Bölümü*
*baharrsarimehmetoglu@gmail.com*

---

## Özet

E-ticaret platformlarında kullanıcı davranışından satın alma niyeti tahmin etmek, kişiselleştirme ve hedefleme için kritik bir problemdir. Bu çalışmada Ekim 2019 – Şubat 2020 arasında toplanan ~16M event ve 3.6M oturum içeren bir e-ticaret veri seti kullanılarak iki aşamalı bir tahmin sistemi önerilmiştir. Birinci aşama binary sınıflandırıcı, bir oturumun satın alma ile sonuçlanıp sonuçlanmayacağını tahmin etmektedir. İkinci aşama çok sınıflı sınıflandırıcı, oturumun beş ürün kategorisinden hangisiyle ilişkili olduğunu belirlemektedir. Event kayıtları oturum seviyesine toplanmış ve her oturum için 14 davranışsal özellik türetilmiştir. Sınıf dengesizliği `class_weight='balanced'` ve `scale_pos_weight` stratejileriyle yönetilmiştir. XGBoost modeli, binary görevde ROC-AUC 0.94 ve çok sınıflı görevde Macro F1 0.85 başarısı sağlamıştır. Zamansal doğrulama testi, sonuçların zaman içinde stabil kaldığını göstermiştir.

**Anahtar Kelimeler**: e-ticaret, satın alma tahmini, oturum analizi, XGBoost, sınıf dengesizliği, zamansal doğrulama.

---

## I. Giriş

E-ticaret trafiği, kullanıcıların ürünleri inceleyip çoğu zaman hiçbir şey almadan çıktığı yoğun bir etkileşim akışıdır. Bu akışta satın alma niyetini erken tespit etmek, öneri sistemleri ve indirim hedeflemesi için değerlidir.

Bu çalışma iki probleme birlikte yanıt arar. **Birinci problem**: Bir oturum satın alma ile sonuçlanacak mı? (İkili sınıflandırma). **İkinci problem**: Sonuçlanacaksa hangi ürün kategorisindendir? (5 sınıflı sınıflandırma). İki problem farklı veri alt kümelerinde çalışır. Bu nedenle tek bir çoklu çıktı modeli yerine iki ayrı model tercih edilmiştir.

İki ana zorluk vardır. Birincisi, satın alma olaylarının tüm oturumlara oranı sadece **%3.32**'dir. Klasik doğruluk metriği bu dengesizlik altında yanıltıcıdır. İkincisi, ham `category_code` alanı ~%98 oranında boştur; kategoriyi doldurmadan çok sınıflı model kurulamaz.

Bu çalışmanın katkıları şunlardır:

1. Event seviyesinden oturum seviyesine temiz bir toplama (aggregation) boru hattı tasarlanmıştır.
2. Hedef sızıntısı (target leakage) teşhis edilmiş ve feature hesaplamasından purchase event'leri dışlanarak düzeltilmiştir.
3. İki aşamalı model yapısı ile binary (satın alma) ve multi-class (kategori) tahminleri ayrı ayrı optimize edilmiştir.
4. Random split'e ek olarak zamansal split ile production realism doğrulanmıştır.

---

## II. Veri Seti

Kaggle üzerinden halka açık olan `mkechinov/ecommerce-events-history-in-cosmetics-shop` veri seti kullanılmıştır [1]. Veri, Ekim 2019 – Şubat 2020 arasındaki dört aylık CSV dosyasından oluşmaktadır. Toplam yaklaşık **16 milyon** event kaydı içerir.

Her event kaydı zaman damgası, event tipi (`view`, `cart`, `remove_from_cart`, `purchase`), ürün kimliği, kategori kimliği, kategori kodu, marka, fiyat, kullanıcı kimliği ve oturum kimliği (`user_session`) alanlarını içerir. Modelleme birimi olarak **oturum** seçilmiştir: bir kullanıcı oturumu, tek bir tarama sürecindeki etkileşimlerin bütününü temsil eder.

Event'ler oturum seviyesine toplandığında **3.599.496** benzersiz oturum elde edilmiştir. Bu oturumların yalnızca **%3.32**'si en az bir purchase event içerir. Dolayısıyla veri ciddi biçimde dengesizdir (Tablo 1).

**Kategori kurtarma**: Ham `category_code` ~%98.3 NULL olmasına rağmen `category_id` neredeyse her kayıtta mevcuttur. `category_id` her kayıtta mevcut olduğu için, `category_code` dolu olan kayıtlardan bir eşleme tablosu oluşturulmuş ve bu tablo boş olan kayıtlara uygulanmıştır. Bu adım olmadan çok sınıflı model için kullanılabilir veri %1.7 seviyesine düşerdi.

**Tablo 1**: Veri Seti İstatistikleri

| Metrik | Değer |
|---|---:|
| Toplam event | ~16.000.000 |
| Toplam oturum | 3.599.496 |
| Oturum başına satın alma oranı | 0.0332 |
| Kategori bilgisi kurtarılan oturum | 131.297 |
| Ana kategori sayısı | 5 |

**Tablo 2**: Ana Kategori Dağılımı (kategorili 131K oturum üzerinden)

| Kategori | Oturum | Pay |
|---|---:|---:|
| appliances | 59.522 | 45.3% |
| furniture | 24.462 | 18.6% |
| apparel | 19.037 | 14.5% |
| stationery | 15.977 | 12.2% |
| accessories | 12.299 | 9.4% |

---

## III. Metodoloji

### A. Event → Oturum Toplama

Modelleme birimi oturumdur. Her oturum için event türlerinin sayısı, ürün çeşitliliği, fiyat istatistikleri ve oturum süresi hesaplanmıştır.

### B. Özellik Mühendisliği

Her oturum için 14 davranışsal özellik türetilmiştir (Tablo 3). Bu özellikler üç kategoriye ayrılabilir: (i) aktivite sayımları, (ii) fiyat istatistikleri, (iii) çeşitlilik/keşif ölçüleri. `cart_to_view_ratio` ve `remove_to_cart_ratio` gibi oran özellikleri, dönüşüm ve tereddüt sinyallerini yakalar. Sıfıra bölme durumu `np.where` koruması ile ele alınmıştır.

**Tablo 3**: Oturum Seviyesi Özellikler

| Özellik | Yorum |
|---|---|
| `n_view`, `n_cart`, `n_remove`, `n_events` | Aktivite yoğunluğu |
| `cart_to_view_ratio` | Dönüşüm oranı |
| `remove_to_cart_ratio` | Tereddüt |
| `avg/max/min_price`, `price_std` | Fiyat profili |
| `unique_products`, `unique_categories` | Keşif genişliği |
| `has_brand` | Markalı ürüne maruz kalma |
| `session_duration_sec` | Oturum süresi |

### C. Hedef Sızıntısı Kontrolü

İlk deneylerde Logistic Regression modeli F1 = 0.9999 gibi gerçek dışı bir sonuç verdi. Araştırma, bunun `n_events` sayımının `purchase` event'lerini de içermesinden kaynaklandığını gösterdi: `purchased = 1` olan oturumlarda `n_events > n_view + n_cart + n_remove` ilişkisi her zaman sağlanıyordu. Bu **deterministik bir hedef sinyali** oluşturuyordu.

Düzeltme olarak tüm özellikler **yalnızca purchase olmayan event'ler** üzerinden hesaplanmıştır. Hedef değişken (`purchased`) ise tüm event'lerden türetilmiştir. Bu düzeltmeden sonra her oturumda `n_events == n_view + n_cart + n_remove` invariantı sağlanmıştır. Hiçbir özelliğin tek başına üniversal AUC değeri 0.835'i aşmamaktadır (makul sinyal gücü, leakage yok).

### D. Sınıf Dengesizliği Stratejisi

Her iki modelde de `class_weight='balanced'` (LogReg) ve `scale_pos_weight` (XGBoost binary) stratejileri kullanılmıştır. Çok sınıflı XGBoost için `compute_sample_weight('balanced')` ile örnek bazlı ağırlıklar uygulanmıştır. SMOTE gibi sentetik örnekleme **bilinçli olarak kullanılmamıştır**. Gerekçe: (i) veri hacmi (3.6M oturum) yeterlidir, (ii) sentetik örnekler test sızıntısı riski taşır, (iii) ağırlıklandırma daha basit ve şeffaftır.

### E. Model Seçimi

**Primary model**: XGBoost [2]. Gradient-boosted karar ağaçları, sayısal ve ölçeksiz özellikler üzerinde non-lineer etkileşimleri yakalar. Ayrıca outlier'lara karşı robust, ağırlıklandırma desteği nativedir. Alternatif olarak Random Forest [3] da değerlendirilmiştir ancak XGBoost'un native imbalance handling (`scale_pos_weight`) ve gradient-based optimization avantajları nedeniyle primary model olarak seçilmiştir.

**Baseline**: Logistic Regression (standardize edilmiş girdiler ile). Doğrusal sınırlı baz model olarak kıyaslama sağlar.

### F. Değerlendirme Protokolü

Stratified 80/20 split, `random_state=42`. Binary için öncelikli metrik **F1** ve **ROC-AUC**. Çok sınıflı için **Macro F1** ve **Weighted F1**. Klasik doğruluk bilinçli olarak raporlanmamıştır: %3.3 pozitif sınıfta "hep 0 tahmin et" modeli %96.7 doğruluk verir ama pratik değeri yoktur.

---

## IV. Deneyler ve Sonuçlar

### A. Binary Sınıflandırma Sonuçları

**Tablo 4**: Binary Model Karşılaştırması (random split, test n=719.900)

| Model | F1 | ROC-AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.256 | 0.862 | 0.158 | 0.661 |
| **XGBoost** | **0.297** | **0.940** | 0.177 | 0.923 |

XGBoost, LogReg baseline'ına göre F1'de +0.041 ve ROC-AUC'de +0.078 kazanç sağlamıştır. Precision-Recall eğrisinin altındaki alan (Average Precision) **0.441**'dir. Baz oran 0.033 olduğu için bu, rastgele tahmine göre yaklaşık **13 kat** iyileşmeye karşılık gelir.

Şekil 1, ROC ve PR eğrilerini göstermektedir. Şekil 2, confusion matrix'i içermektedir. Model, 23.885 gerçek satın almadan 22.046'sını yakalar (Recall 0.923) ancak 102.696 false positive üretir. Bu, `scale_pos_weight` stratejisinin **recall'u önceleyen** doğası ile tutarlıdır.

*(Şekil 1: `results/figures/binary_roc.png` ve `binary_pr.png`)*
*(Şekil 2: `results/figures/binary_confusion.png`)*

### B. Çok Sınıflı Sınıflandırma Sonuçları

**Tablo 5**: Çok Sınıflı Model Karşılaştırması (test n=26.260)

| Model | Macro F1 | Weighted F1 |
|---|---:|---:|
| Logistic Regression | 0.556 | 0.617 |
| **XGBoost** | **0.855** | **0.878** |

XGBoost, LogReg baseline'ına kıyasla Macro F1'de **+0.30** kazanç sağlamıştır. Bu fark, kategori ayırımı için doğrusal olmayan etkileşimlerin kritik olduğunu göstermektedir.

**Tablo 6**: Sınıf Bazında F1 (XGBoost)

| Kategori | F1 | Yorum |
|---|---:|---|
| appliances | 0.927 | Fiyat/marka imzası net |
| furniture | 0.883 | Yüksek fiyat segmenti |
| accessories | 0.866 | Azınlık ama ayrılabilir |
| stationery | 0.846 | Düşük fiyat, stabil |
| apparel | 0.752 | Fiyat örtüşmesi (en zor) |

Şekil 3, normalize confusion matrix'i göstermektedir. En büyük karışıklık `appliances → apparel` yönündedir (696 örnek). Bunun nedeni muhtemelen apparel kategorisinin fiyat dağılımının diğer kategorilerle örtüşmesidir.

*(Şekil 3: `results/figures/multiclass_confusion_normalized.png`)*

### C. 5-Fold Cross-Validation Kararlılığı

Tablo 7, 5-fold stratified CV sonuçlarını özetler. Standart sapmalar son derece küçüktür (Binary için σ(F1) ≈ 0.001, Multi için σ(Macro F1) ≈ 0.001). Bu, sonuçların tek-split tesadüfünden bağımsız, istikrarlı olduğunu doğrular.

**Tablo 7**: 5-Fold CV Sonuçları (ortalama ± standart sapma)

| Görev | Metrik | Değer |
|---|---|---:|
| Binary | F1 | 0.2970 ± 0.0007 |
| Binary | ROC-AUC | 0.9409 ± 0.0006 |
| Multi-class | Macro F1 | 0.8536 ± 0.0014 |
| Multi-class | Weighted F1 | 0.8757 ± 0.0014 |

### D. Zamansal Doğrulama

Random split, i.i.d. varsayımı yapar. Gerçek dağıtımda model geçmişten geleceği tahmin eder. Bu nedenle ek bir doğrulama olarak **zamansal split** uygulanmıştır: Train = Ekim + Aralık 2019 + Ocak 2020. Test = Şubat 2020.

**Tablo 8**: Random vs Zamansal Karşılaştırma

| Metrik | Random | Zamansal | Δ |
|---|---:|---:|---:|
| Binary F1 | 0.297 | 0.283 | −0.013 |
| Binary ROC-AUC | 0.940 | 0.939 | **−0.001** |
| Multi Macro F1 | 0.855 | 0.819 | −0.036 |
| Multi Weighted F1 | 0.878 | 0.844 | −0.034 |

Binary ROC-AUC neredeyse hiç düşmemiştir. Bu, modelin oturumları satın alma olasılığına göre **zamana dayanıklı** biçimde sıraladığı anlamına gelir. F1'deki küçük düşüş, threshold 0.5'te kalibrasyondan kaynaklanır. Çok sınıflı Macro F1'deki −0.036'lık düşüş **ılımlı bir dağılım kayması**dır; yapısal bir drift yoktur.

En büyük zamansal kayıp accessories kategorisinde yaşanmıştır (Δ −0.076): azınlık sınıfları genelde her türlü dağılım kaymasına en duyarlı olanlardır.

---

## V. Tartışma

**Precision-Recall dengesi**. XGBoost binary model, threshold 0.5'te Recall = 0.923 ama Precision = 0.177 vermektedir. Bu, `scale_pos_weight` stratejisinin doğrudan bir sonucudur: azınlık sınıfını üst sınıra itecek şekilde pozitifleri agresif öngörmektedir. Gerçek dağıtımda threshold bir operasyonel parametre olarak ayarlanabilir. PR eğrisi, deployment aşamasında maliyet-fayda analiziyle threshold kararı için yeterli bilgiyi sağlar.

**Hedef sızıntısı deneyimi**. Başlangıçtaki F1 = 0.9999 sonucu, "çok iyi görünüyorsa muhtemelen leakage vardır" uyarısının somut bir örneğidir. Tüm davranış özelliklerinin sadece purchase **olmayan** event'lerden türetilmesi, probleme epistemolojik olarak da uygun bir kurgudur: "satın alma-öncesi davranıştan satın almayı tahmin et".

**Zamansal kararlılık**. 4 ay gibi kısa bir dilimde güçlü bir concept drift beklenmez. Sonuçlarımız da bu beklentiyi doğrular: Binary ROC-AUC neredeyse değişmez (Δ −0.001), Multi Macro F1 %4.2 düşer. Yine de deployment'tan sonra düzenli olarak yeniden eğitim (retraining) önerilir.

**Kısıtlar**. (i) Oturum seviyesi modelleme, kullanıcı kimliğini bağlamı zamansal olarak kullanmaz: aynı kullanıcının geçmişteki davranışı mevcut oturuma yansıtılmaz. (ii) 14 özellik davranış sinyallerinin çekirdeğini yakalar ama zaman-arası (time-of-day, haftanın günü) sinyalleri dışarıda bırakır. (iii) Çok sınıflı modelin eğitim kümesi, `category_id` eşlemesinde var olan oturumlarla sınırlıdır; **satın alma oturumlarının %90'ı kategori bilgisi olmadığı için bu modelin eğitim setinde yoktur**. Dolayısıyla Model 2, "satın alma kategorisi"nden ziyade "oturumun baskın ilgi kategorisi"ni tahmin etmektedir.

---

## VI. Sonuç

Bu çalışma, büyük ölçekli bir e-ticaret veri seti üzerinde oturum seviyesi satın alma niyeti ve kategori tahmini için iki aşamalı bir pipeline sunmuştur. 14 davranışsal özellik ve XGBoost tabanlı modeller ile binary görevde ROC-AUC 0.94, çok sınıflı görevde Macro F1 0.85 başarısı elde edilmiştir. Zamansal doğrulama ve 5-fold çapraz doğrulama ile sonuçların hem istatistiksel hem zamansal olarak kararlı olduğu gösterilmiştir. Başlangıçta yakalanan hedef sızıntısı, özellik mühendisliği aşamasında titiz doğrulamanın önemini vurgulamıştır.

---

## Referanslar

[1] M. Kechinov, *eCommerce events history in cosmetics shop*, Kaggle, 2020. [Çevrimiçi]. Erişim: https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop

[2] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.

[3] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[4] H. He and E. A. Garcia, "Learning from Imbalanced Data," *IEEE Transactions on Knowledge and Data Engineering*, vol. 21, no. 9, pp. 1263–1284, 2009.

[5] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[6] J. A. Hanley and B. J. McNeil, "The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve," *Radiology*, vol. 143, no. 1, pp. 29–36, 1982.

[7] J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, and A. Bouchachia, "A Survey on Concept Drift Adaptation," *ACM Computing Surveys*, vol. 46, no. 4, pp. 1–37, 2014.
