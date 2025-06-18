# 🔓 CAPTCHA Breaker

<div align="center">

![Python](https://img.shields.io/badge/python-v3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.10+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![GPU](https://img.shields.io/badge/GPU-Accelerated-brightgreen.svg)

*Breaking CAPTCHAs with 98%+ accuracy using Convolutional Recurrent Neural Networks and CTC loss*
</div>

---

## 🌟 Headlines

- **🎯 98%+ character accuracy** on synthetic CAPTCHAs
- **⚡ GPU-accelerated training** with mixed precision
- **🔄 End-to-end pipeline** from data generation to deployment
- **🧠 Full CTC evaluation** 
- **🎨 Synthetic data generation** - no manual labeling required

## 🚀 Quick Start

Get up and running in minutes:

### 1️⃣ Install Dependencies
```bash
pip install tensorflow tensorflow-addons omegaconf hydra-core tqdm matplotlib captcha pillow
```

### 2️⃣ Generate Training Data
```bash
python generate_synthetic_dataset.py -n 20000
```
*Creates 20,000 synthetic CAPTCHA images like `ABC123_001.png`*

### 3️⃣ Train the Model
```bash
python train.py
```
*Auto-detects GPU based on TensorFlow, trains with mixed precision, saves best model*

### 4️⃣ Test Your Model
```bash
python evaluate.py
python predict.py path/to/your/captcha.png
```


---

## 📖 How It Works

### 🧠 The Architecture

Our CRNN (Convolutional Recurrent Neural Network) combines three powerful components:

```
📸 CAPTCHA Image → 🔍 CNN Feature Extractor → 🔄 Bidirectional LSTM → 📝 CTC Decoder → ✨ Text Output
```

#### 1. **CNN Backbone** 🔍
- ResNet-inspired feature extractor
- Batch normalization + ReLU activation
- Progressive max pooling for spatial reduction
- Converts images to rich feature representations

#### 2. **Sequence Modeling** 🔄
- **Bidirectional LSTM layers** capture left-to-right AND right-to-left context
- Handles variable-length sequences automatically
- Dropout prevents overfitting

#### 3. **CTC Magic** ✨
- **Connectionist Temporal Classification** eliminates need for character-level alignment
- Handles variable-length outputs elegantly
- Proper decoding removes duplicates and blank tokens

### 🎯 Why CTC is Crucial

**❌ Traditional Approach:**
```
Requires: [A][B][C][1][2][3] ← Exact alignment needed
```

**✅ CTC Approach:**
```
Handles: [A][A][_][B][C][_][1][2][3][3][_] ← Automatic alignment
         ↓ CTC Decoding ↓
Output:  ABC123
```

---

## 🎯 Results & Performance

### 📊 Accuracy Metrics

| Metric | Score | Description |
|--------|-------|-------------|
| **Character Accuracy** | 99.3% | Individual character recognition |
| **Sequence Accuracy** | 97.9% | Complete CAPTCHA solved correctly |
| **Training Time** | <1 hour | On RTX 3050 (mixed precision) |
| **Inference Speed** | ~10ms | Per image on GPU |

### 🏆 Benchmarks

**Training Performance:**
- **Dataset:** 20,000 synthetic CAPTCHAs
- **Convergence:** 20-40 epochs (early stopping)
- **Memory Usage:** ~2GB GPU memory
- **Speed:** 40% faster with mixed precision

### 📈 Learning Curves

The model typically shows:
- Rapid initial learning (epochs 1-10)
- Gradual improvement (epochs 10-30)
- Convergence with early stopping

---

### 🐛 Troubleshooting

**Common Issues:**

1. **CUDA out of memory:**
   ```yaml
   # Reduce batch size in config.yaml
   batch_size: 64  # or 32
   ```

2. **Mixed precision errors:**
   ```yaml
   # Disable for older GPUs
   mixed_precision: false
   ```

---

## 📜 License

This project is licensed under the **MIT License** - see [LICENSE.txt](LICENSE.txt) for details.

**TL;DR:** Use it freely for educational and commercial purposes! 🎉
