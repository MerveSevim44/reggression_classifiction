# reggression_classifiction
# Regression & Classification Evaluation (Çalışma Amaçlı)

Bu proje, **regresyon** ve **sınıflandırma** modellerinin temel değerlendirme metriklerini öğrenmek ve uygulamak amacıyla hazırlanmıştır. Proje tamamen **çalışma ve pratik amaçlıdır**, gerçek bir üretim veya araştırma projesi değildir.

## İçerik

Proje iki ana bölümden oluşmaktadır:

### 1. Regression Evaluation
Dosya: `evaluation_regression.py`

Bu bölümde:
- Basit bir doğrusal regresyon modeli kurulmuştur.
- Kullanılan metrikler:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
- Sonuçlar grafik üzerinde görselleştirilmiştir.

Kullanılan kütüphaneler:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

### 2. Classification Evaluation
Dosya: `evaluation_classification.py`

Bu bölümde:
- Logistic Regression modeli kurulmuştur.
- Kullanılan metrikler:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Classification Report

Ayrıca sınıf dengesizliği durumunun metrikler üzerindeki etkisi yorumlanmıştır.

Kullanılan kütüphaneler:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## Amaç

Bu projenin amacı:
- Regresyon ve sınıflandırma modellerini değerlendirmeyi öğrenmek
- Temel performans metriklerini anlamak
- Python ve scikit-learn pratiği yapmak

Bu proje **öğrenme ve deneme amaçlıdır**.

---

## Çalıştırma

Gerekli kütüphaneleri yüklemek için:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
