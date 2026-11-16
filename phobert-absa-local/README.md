# PhoBERT Smartphone ABSA — Hướng dẫn cài đặt và vận hành trên Windows

Hệ thống phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis - ABSA) từ đánh giá smartphone tiếng Việt, sử dụng mô hình PhoBERT đã fine-tune kết hợp FastAPI backend và React/Vite/TypeScript frontend.

---

## 1. Cấu trúc thư mục

```text
phobert-absa-local/
├── backend/
│   ├── app.py             # FastAPI backend server
│   ├── test_api.py        # Test suite kiểm thử tự động
│   └── requirements.txt   # Danh sách thư viện Python
├── frontend/
│   ├── src/
│   │   ├── main.tsx       # Ứng dụng React ABSA
│   │   └── style.css      # Giao diện responsive tối ưu
│   ├── package.json
│   └── vite.config.ts
├── model/                 # Thư mục checkpoint PhoBERT fine-tuned
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── added_tokens.json
│   ├── bpe.codes
│   └── vocab.txt
└── .venv/                 # Môi trường ảo Python 3.11
```

---

## 2. Chuẩn bị Checkpoint mô hình

Các file checkpoint fine-tune từ Google Colab được đặt trong `model/`:
- `config.json` (kiến trúc `RobertaForSequenceClassification`, `vocab_size: 64003`, `num_labels: 3`)
- `model.safetensors` (trọng số fine-tuned)
- `tokenizer_config.json` (chứa special tokens `[ASPECT]` và `[TEXT]` dưới khóa chuẩn `additional_special_tokens`)
- `vocab.txt`, `bpe.codes`, `added_tokens.json`

> **Lưu ý tương thích:** Trong Transformers 4.45+, special tokens bổ sung phải được định danh dưới khóa `additional_special_tokens: ["[ASPECT]", "[TEXT]"]` trong `tokenizer_config.json` thay vì `extra_special_tokens` (vốn là dictionary ánh xạ thuộc tính trong các phiên bản mới).

---

## 3. Khởi chạy Backend (FastAPI)

Mở **PowerShell Terminal 1** tại thư mục `phobert-absa-local`:

```powershell
# 1. Kích hoạt môi trường ảo Python 3.11 (hoặc tạo mới nếu cần)
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt các gói phụ thuộc (nếu chưa cài)
python -m pip install -r backend\requirements.txt

# 3. Khởi động FastAPI server
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

- **Swagger API Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Kiểm tra Health:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

### Chạy bộ kiểm thử tự động (Test Suite)

Chạy script test toàn diện kiểm tra tokenizer, model weights, validation và các test cases:

```powershell
$env:PYTHONPATH = "d:\HOC_2026_2027\XLNNTN\PHOBERT_SMARTPHONE\phobert-absa-local"
.\.venv\Scripts\python.exe -m backend.test_api
```

---

## 4. Khởi chạy Frontend (React + Vite)

Mở **PowerShell Terminal 2** tại thư mục `phobert-absa-local\frontend`:

```powershell
cd frontend
npm install
npm run dev
```

- Mở trình duyệt tại: [http://127.0.0.1:5173](http://127.0.0.1:5173)

---

## 5. Danh sách API Endpoints

### 1. `GET /api/health`
Kiểm tra chi tiết trạng thái hoạt động của server và checkpoint:
```json
{
  "status": "ok",
  "checkpoint_exists": true,
  "model_loaded": true,
  "model_ready": true,
  "device": "cpu"
}
```

### 2. `GET /api/aspects`
Trả về 10 khía cạnh smartphone được mô hình hỗ trợ:
```json
{
  "aspects": [
    "GENERAL", "PERFORMANCE", "BATTERY", "FEATURES", "CAMERA",
    "PRICE", "SER&ACC", "DESIGN", "SCREEN", "STORAGE"
  ]
}
```

### 3. `POST /api/analyze`
Phân tích cảm xúc theo khía cạnh được chọn.

**Request Body:**
```json
{
  "comment": "Camera chụp ảnh rất đẹp nhưng pin nhanh hết.",
  "aspects": ["CAMERA", "BATTERY"]
}
```

**Response (HTTP 200):**
```json
{
  "comment": "Camera chụp ảnh rất đẹp nhưng pin nhanh hết.",
  "results": [
    {
      "aspect": "CAMERA",
      "sentiment": "Positive",
      "confidence": 0.9927,
      "scores": {
        "Negative": 0.0015,
        "Neutral": 0.0059,
        "Positive": 0.9927
      }
    },
    {
      "aspect": "BATTERY",
      "sentiment": "Negative",
      "confidence": 0.9911,
      "scores": {
        "Negative": 0.9911,
        "Neutral": 0.005,
        "Positive": 0.0039
      }
    }
  ],
  "note": "Kết quả chỉ đánh giá các khía cạnh được chọn; không xác định khía cạnh có được nhắc đến hay không."
}
```

---

## 6. Giới hạn phương pháp luận quan trọng

1. **Không có nhãn `Not Mentioned`:** Tập dữ liệu huấn luyện chỉ gán nhãn cảm xúc (Negative: 0, Neutral: 1, Positive: 2) cho các khía cạnh được cung cấp, không phân loại khía cạnh có xuất hiện hay không. Nếu chọn một khía cạnh không có trong câu, mô hình vẫn sẽ phân loại cảm xúc cho khía cạnh đó. Người dùng cần chủ động chọn khía cạnh cần quan tâm.
2. **Định dạng đầu vào (Input Pipeline):** Mô hình sử dụng tiền xử lý tiếng Việt nguyên bản, định dạng chuỗi `[ASPECT] {aspect} [TEXT] {comment}` với độ dài tối đa `max_length = 128` khớp chính xác với notebook huấn luyện.
