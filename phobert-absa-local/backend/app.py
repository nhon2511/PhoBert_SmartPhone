from functools import lru_cache
from pathlib import Path
import logging
import os

import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('phobert_absa')

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv('MODEL_DIR', str(ROOT / 'model')))
MAX_LENGTH = 128
ASPECTS = ['GENERAL', 'PERFORMANCE', 'BATTERY', 'FEATURES', 'CAMERA',
           'PRICE', 'SER&ACC', 'DESIGN', 'SCREEN', 'STORAGE']
LABELS = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

app = FastAPI(title='PhoBERT Smartphone ABSA', version='1.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:5173', 'http://127.0.0.1:5173'],
    allow_credentials=False,
    allow_methods=['GET', 'POST'],
    allow_headers=['Content-Type'],
)

class AnalyzeRequest(BaseModel):
    comment: str = Field(min_length=1, max_length=10000)
    aspects: list[str] = Field(min_length=1, max_length=10)

@lru_cache(maxsize=1)
def get_model():
    config_file = MODEL_DIR / 'config.json'
    weights_safetensors = MODEL_DIR / 'model.safetensors'
    weights_bin = MODEL_DIR / 'pytorch_model.bin'
    
    if not config_file.exists():
        raise RuntimeError(f'Không tìm thấy file config.json ở {MODEL_DIR}. Hãy export model từ Colab.')
    if not (weights_safetensors.exists() or weights_bin.exists()):
        raise RuntimeError(f'Không tìm thấy file trọng số (model.safetensors hoặc pytorch_model.bin) ở {MODEL_DIR}.')

    logger.info('Đang tải tokenizer từ %s...', MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    
    logger.info('Đang tải weights mô hình từ %s...', MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR), local_files_only=True)
    
    if model.config.num_labels != 3:
        raise RuntimeError(f'Mô hình có {model.config.num_labels} nhãn, không phải 3 nhãn cảm xúc (Negative, Neutral, Positive).')
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise RuntimeError(
            f'Tokenizer ({len(tokenizer)}) và model ({model.get_input_embeddings().num_embeddings}) '
            'có kích thước vocabulary khác nhau.'
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    logger.info('Tải mô hình thành công lên thiết bị %s (vocab: %d, nhãn: %d)', device, len(tokenizer), model.config.num_labels)
    return tokenizer, model, device

@app.get('/api/health')
def health(preload: bool = Query(default=False, description='Tự động nạp model vào RAM nếu chưa nạp')):
    checkpoint_exists = (MODEL_DIR / 'config.json').exists() and (
        (MODEL_DIR / 'model.safetensors').exists() or (MODEL_DIR / 'pytorch_model.bin').exists()
    )
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if preload and checkpoint_exists and get_model.cache_info().currsize == 0:
        try:
            get_model()
        except Exception as e:
            logger.exception('Lỗi khi preload mô hình: %s', e)

    model_loaded = get_model.cache_info().currsize > 0
    model_ready = model_loaded and checkpoint_exists
    
    return {
        'status': 'ok',
        'checkpoint_exists': checkpoint_exists,
        'model_loaded': model_loaded,
        'model_ready': model_ready,
        'device': device_name
    }

@app.get('/api/aspects')
def aspects():
    return {'aspects': ASPECTS}

@app.post('/api/analyze')
def analyze(req: AnalyzeRequest):
    cleaned_comment = req.comment.strip()
    if not cleaned_comment:
        raise HTTPException(status_code=422, detail='Nội dung bình luận không được để trống.')

    chosen = list(dict.fromkeys(req.aspects))
    if not chosen:
        raise HTTPException(status_code=422, detail='Vui lòng chọn ít nhất một khía cạnh cần phân tích.')

    invalid = [a for a in chosen if a not in ASPECTS]
    if invalid:
        raise HTTPException(status_code=422, detail=f'Khía cạnh không hợp lệ: {invalid}')

    try:
        tokenizer, model, device = get_model()
    except Exception as e:
        logger.exception('Lỗi khi tải hoặc chuẩn bị mô hình: %s', e)
        raise HTTPException(
            status_code=503,
            detail='Mô hình chưa sẵn sàng để suy luận. Vui lòng kiểm tra lại trạng thái server.'
        ) from e

    try:
        texts = [f'[ASPECT] {aspect} [TEXT] {cleaned_comment}' for aspect in chosen]
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        
        with torch.inference_mode():
            logits = model(**tokens).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()

        results = []
        for aspect, probs in zip(chosen, probabilities):
            idx = max(range(3), key=lambda i: probs[i])
            results.append({
                'aspect': aspect,
                'sentiment': LABELS[idx],
                'confidence': round(probs[idx], 4),
                'scores': {LABELS[i]: round(probs[i], 4) for i in range(3)},
            })
        return {
            'comment': req.comment,
            'results': results,
            'note': 'Kết quả chỉ đánh giá các khía cạnh được chọn; không xác định khía cạnh có được nhắc đến hay không.'
        }
    except Exception as e:
        logger.exception('Lỗi trong quá trình suy luận mô hình: %s', e)
        raise HTTPException(
            status_code=500,
            detail='Đã xảy ra lỗi nội bộ trong quá trình phân tích bình luận.'
        ) from e
