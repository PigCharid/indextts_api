# app/main.py
import os
import uuid
import json
import time
import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler
from typing import Optional, List, Tuple, Union
from contextlib import asynccontextmanager

import tempfile
import requests
from urllib.parse import urlparse

from fastapi import FastAPI, UploadFile, HTTPException, Request
from fastapi import File, Form, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import UploadFile as StarletteUploadFile  # 兼容

# 允许 https://IP，关闭 TLS 校验告警
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# IndexTTS2
from indextts.infer_v2 import IndexTTS2

APP_TITLE = "IndexTTS2 API"
APP_DESC = "Zero-shot TTS with emotion control (FastAPI wrapper)."
APP_VERSION = "0.1.0"

# ===== 环境变量 =====
CHECKPOINT_DIR = os.getenv("INDEXTTS_CHECKPOINTS", "checkpoints")
CFG_PATH = os.getenv("INDEXTTS_CONFIG", os.path.join(CHECKPOINT_DIR, "config.yaml"))
OUTPUT_DIR = os.getenv("INDEXTTS_OUTPUT_DIR", "outputs")
USE_FP16 = os.getenv("INDEXTTS_FP16", "true").lower() == "true"
USE_DS = os.getenv("INDEXTTS_DEEPSPEED", "false").lower() == "true"
USE_CUDA_KERNEL = os.getenv("INDEXTTS_CUDA_KERNEL", "false").lower() == "true"
LOG_DIR = os.getenv("INDEXTTS_LOG_DIR", "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("/tmp/uploads", exist_ok=True)

# ===== 日志：始终创建一个文件 handler，并复用它到所有 logger =====
LOG_FILE = os.path.join(LOG_DIR, "indextts.log")

def _build_file_handler() -> TimedRotatingFileHandler:
    fh = TimedRotatingFileHandler(
        LOG_FILE, when="D", interval=1, backupCount=14, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    return fh

def _setup_logging() -> Tuple[logging.Logger, logging.Handler, logging.Handler]:
    file_handler = _build_file_handler()
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    # 主业务 logger
    app_logger = logging.getLogger("indextts")
    app_logger.setLevel(logging.INFO)
    # 避免重复添加
    if not any(isinstance(h, TimedRotatingFileHandler) and h.baseFilename == os.path.abspath(LOG_FILE)
               for h in app_logger.handlers):
        app_logger.addHandler(file_handler)
        app_logger.addHandler(console_handler)

    # 将 uvicorn 的日志也打到同一文件
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        l = logging.getLogger(name)
        l.setLevel(logging.INFO)
        if not any(isinstance(h, TimedRotatingFileHandler) and h.baseFilename == os.path.abspath(LOG_FILE)
                   for h in l.handlers):
            l.addHandler(file_handler)
            l.addHandler(console_handler)

    return app_logger, file_handler, console_handler

logger, _file_handler, _console_handler = _setup_logging()

# ===== FastAPI App =====
_model: Optional[IndexTTS2] = None
_infer_lock = asyncio.Lock()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info("🚀 Starting IndexTTS2 API ...")
    try:
        _model = IndexTTS2(
            cfg_path=CFG_PATH,
            model_dir=CHECKPOINT_DIR,
            use_fp16=USE_FP16,
            use_cuda_kernel=USE_CUDA_KERNEL,
            use_deepspeed=USE_DS,
        )
        logger.info(
            "✅ Model loaded (device=%s, fp16=%s, ckpt=%s)",
            getattr(_model, "device", None),
            getattr(_model, "use_fp16", None),
            CHECKPOINT_DIR,
        )
    except Exception as e:
        logger.exception("❌ Failed to load IndexTTS2: %s", e)
        raise
    yield
    logger.info("🧹 API shutting down...")

app = FastAPI(title=APP_TITLE, description=APP_DESC, version=APP_VERSION, lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议收敛
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ===== 兜底日志中间件（路由外异常也能写日志）=====
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        try:
            return await call_next(request)
        except Exception:
            logger.exception("UNHANDLED ERROR | %s %s", request.method, request.url)
            raise

app.add_middleware(LoggingMiddleware)

# ===== 全局异常处理（统一落日志）=====
from fastapi import HTTPException
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(
        "HTTPException | %s %s -> %s (%s)",
        request.method, request.url, exc.detail, exc.status_code
    )
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

from fastapi.exceptions import RequestValidationError
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
    except Exception:
        body = b""
    logger.warning(
        "ValidationError | %s %s | body=%s | errors=%s",
        request.method, request.url,
        (body[:1024].decode(errors="ignore") if body else ""),
        exc.errors()
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Exception | %s %s | %r", request.method, request.url, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# ===== UploadFile 兼容判断元组（少量工具函数会用到）=====
FILE_TYPES = (UploadFile, StarletteUploadFile)

# ===== Pydantic (JSON 接口用) =====
class SynthesizeJSONReq(BaseModel):
    text: str
    spk_audio_path: Optional[str] = None   # 本地路径 或 http(s):// URL
    emo_audio_path: Optional[str] = None   # 本地路径 或 http(s):// URL
    emo_alpha: float = 1.0
    emo_vector: Optional[List[float]] = None
    use_random: bool = False
    use_emo_text: bool = False
    emo_text: Optional[str] = None
    filename: Optional[str] = None

# ===== 工具函数 =====
def _guess_audio_ext(name: str) -> str:
    low = name.lower()
    for cand in (".wav", ".mp3", ".flac", ".m4a", ".ogg"):
        if cand in low or low.endswith(cand):
            return cand
    return ".wav"

def _download_url_to_temp(url: str) -> str:
    """
    下载 URL 到临时文件（/tmp），关闭 TLS 校验以支持 https://IP。
    返回本地路径。调用方负责删除。
    """
    ext = _guess_audio_ext(url)
    tmp = tempfile.NamedTemporaryFile(prefix="indextts_", suffix=ext, delete=False, dir="/tmp")
    tmp_path = tmp.name
    tmp.close()
    t0 = time.perf_counter()
    try:
        with requests.get(url, timeout=60, stream=True, verify=False) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        logger.info("Downloaded: %s -> %s (%.2fms)", url, tmp_path, (time.perf_counter()-t0)*1000)
        return tmp_path
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        logger.exception("Download failed from %s: %s", url, e)
        raise HTTPException(status_code=400, detail=f"下载音频失败: {e}")

def _write_upload_to_temp(upload: UploadFile) -> str:
    """
    将上传的文件写入临时文件。返回路径。调用方负责删除。
    """
    filename = upload.filename or "upload.wav"
    ext = _guess_audio_ext(filename)
    tmp = tempfile.NamedTemporaryFile(prefix="indextts_", suffix=ext, delete=False, dir="/tmp")
    tmp_path = tmp.name
    try:
        with tmp as f:
            while True:
                chunk = upload.file.read(8192)
                if not chunk:
                    break
                f.write(chunk)
        logger.info("Materialized upload '%s' -> %s", filename, tmp_path)
        return tmp_path
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        logger.exception("Failed to materialize upload '%s': %s", filename, e)
        raise HTTPException(status_code=400, detail=f"保存上传音频失败: {e}")

def _is_url(s: str) -> bool:
    try:
        p = urlparse(s)
        return p.scheme in ("http", "https")
    except Exception:
        return False

def _materialize_source(value: Union[str, UploadFile, None]) -> Tuple[Optional[str], Optional[str]]:
    """
    将音频来源（本地路径/URL/UploadFile/None）“实体化”为可供模型读取的本地路径。
    返回 (path, tmp_path)：
        - path: 实体化后的路径（可直接传给模型）
        - tmp_path: 若是临时文件，需要推理完成后删除；否则为 None
    """
    if value is None:
        return None, None
    if isinstance(value, FILE_TYPES):
        p = _write_upload_to_temp(value)
        return p, p
    if isinstance(value, str):
        v = value.strip()
        if _is_url(v):
            p = _download_url_to_temp(v)
            return p, p
        if not os.path.exists(v):
            raise HTTPException(status_code=400, detail=f"本地路径不存在: {v}")
        return v, None
    raise HTTPException(status_code=400, detail="不支持的音频参数类型")

def _cleanup_paths(paths: List[Optional[str]]):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
                logger.info("Cleaned temp: %s", p)
            except Exception as e:
                logger.warning("Failed to remove temp %s: %s", p, e)

def _safe_json(obj) -> str:
    """将对象安全序列化成 JSON，用于日志"""
    def default(o):
        if isinstance(o, FILE_TYPES):
            return {"__upload__": True, "filename": o.filename, "content_type": o.content_type}
        return str(o)
    try:
        return json.dumps(obj, ensure_ascii=False, default=default)
    except Exception:
        return str(obj)

def _infer_to_file(**kwargs):
    t0 = time.perf_counter()
    log_params = {k: (str(v) if not isinstance(v, FILE_TYPES) else f"<UploadFile:{v.filename}>")
                  for k, v in kwargs.items()}
    logger.info("Infer start | params=%s", _safe_json(log_params))
    _model.infer(**kwargs, verbose=True)
    dur = (time.perf_counter() - t0) * 1000
    size = os.path.getsize(kwargs["output_path"]) if os.path.exists(kwargs["output_path"]) else -1
    logger.info("Infer done  | out=%s | size=%d bytes | %.2fms", kwargs["output_path"], size, dur)

def _remove_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("Deleted output file after response: %s", path)
    except Exception as e:
        logger.warning("Failed to delete output %s: %s", path, e)

# ===== 健康与配置 =====
@app.get("/health")
def health():
    ok = _model is not None and os.path.exists(CFG_PATH)
    return {"status": "ok" if ok else "broken"}

@app.get("/config")
def config():
    return {
        "checkpoints": CHECKPOINT_DIR,
        "config_yaml": CFG_PATH,
        "fp16": USE_FP16,
        "deepspeed": USE_DS,
        "cuda_kernel": USE_CUDA_KERNEL,
        "output_dir": OUTPUT_DIR,
    }

# ===== 接口 1：POST /tts（仅 JSON）=====
@app.post("/tts")
async def tts(request: Request, background_tasks: BackgroundTasks):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    ctype = request.headers.get("content-type", "").lower()
    if "application/json" not in ctype:
        raise HTTPException(status_code=415, detail="仅支持 application/json")

    out_path = os.path.join(OUTPUT_DIR, f"indextts2_{uuid.uuid4().hex}.wav")
    temps_to_cleanup: List[Optional[str]] = []

    async with _infer_lock:
        try:
            payload = await request.json()
            logger.info("Received JSON payload: %s", _safe_json(payload))
            req = SynthesizeJSONReq(**payload)

            # 实体化音频来源
            spk_path, spk_tmp = _materialize_source(req.spk_audio_path)
            emo_path, emo_tmp = _materialize_source(req.emo_audio_path)
            temps_to_cleanup += [spk_tmp, emo_tmp]

            if not spk_path:
                raise HTTPException(status_code=422, detail="必须提供说话人参考（本地路径/URL/上传文件）")

            # 自定义输出文件名
            if req.filename:
                out_path = os.path.join(
                    OUTPUT_DIR,
                    req.filename if req.filename.lower().endswith(".wav") else (req.filename + ".wav")
                )

            _infer_to_file(
                spk_audio_prompt=spk_path,
                text=req.text,
                output_path=out_path,
                emo_audio_prompt=emo_path,
                emo_alpha=req.emo_alpha,
                emo_vector=req.emo_vector,
                use_random=req.use_random,
                use_emo_text=req.use_emo_text,
                emo_text=req.emo_text,
            )
        finally:
            # 清理（仅上传/下载的临时材料）
            _cleanup_paths(temps_to_cleanup)

    # 注册返回后删除生成的 out 文件
    background_tasks.add_task(_remove_file, out_path)

    # 用 StreamingResponse 返回并删除
    return StreamingResponse(
        open(out_path, "rb"),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(out_path)}"'},
        background=background_tasks,
    )

# ===== 接口 2：POST /ttsform（仅 multipart/form-data，且只接收文件）=====
@app.post("/ttsform")
async def ttsform(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    spk_audio_prompt: UploadFile = File(...),             # 必填：说话人参考（文件）
    emo_audio_prompt: Optional[UploadFile] = File(None),  # 可选：情感参考（文件）
    emo_alpha: float = Form(1.0),
    use_random: bool = Form(False),
    use_emo_text: bool = Form(False),
    emo_text: Optional[str] = Form(None),
    emo_vector: Optional[str] = Form(None),               # 传字符串：JSON 数组或逗号分隔
    filename: Optional[str] = Form(None),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    out_path = os.path.join(OUTPUT_DIR, f"indextts2_{uuid.uuid4().hex}.wav")
    temps_to_cleanup: List[Optional[str]] = []

    async with _infer_lock:
        try:
            # 记录简化日志
            form_log = {
                "text": text[:1000],
                "emo_alpha": emo_alpha,
                "use_random": use_random,
                "use_emo_text": use_emo_text,
                "emo_text": (emo_text[:200] if emo_text else None),
                "emo_vector": (emo_vector[:200] if emo_vector else None),
                "filename": filename,
                "spk_audio_prompt": {
                    "filename": spk_audio_prompt.filename,
                    "content_type": spk_audio_prompt.content_type,
                },
                "emo_audio_prompt": (
                    {"filename": emo_audio_prompt.filename, "content_type": emo_audio_prompt.content_type}
                    if emo_audio_prompt else None
                ),
            }
            logger.info("Received FORM payload (typed): %s", _safe_json(form_log))

            # 将文件落地为临时路径
            spk_path = _write_upload_to_temp(spk_audio_prompt)
            temps_to_cleanup.append(spk_path)

            emo_path = None
            if emo_audio_prompt is not None:
                emo_path = _write_upload_to_temp(emo_audio_prompt)
                temps_to_cleanup.append(emo_path)

            # 解析 emo_vector（支持 JSON 或 逗号分隔）
            emo_vec = None
            if emo_vector:
                try:
                    emo_vec = json.loads(emo_vector)
                    if not isinstance(emo_vec, list) or not all(isinstance(x, (int, float)) for x in emo_vec):
                        raise ValueError
                    emo_vec = [float(x) for x in emo_vec]
                except Exception:
                    try:
                        parts = [p.strip() for p in emo_vector.split(",") if p.strip()]
                        emo_vec = [float(p) for p in parts]
                    except Exception:
                        raise HTTPException(status_code=400, detail="emo_vector 需为 JSON 数组或逗号分隔数字")

            # 自定义输出文件名
            if filename:
                out_path = os.path.join(
                    OUTPUT_DIR,
                    filename if filename.lower().endswith(".wav") else (filename + ".wav")
                )

            _infer_to_file(
                spk_audio_prompt=spk_path,
                text=text,
                output_path=out_path,
                emo_audio_prompt=emo_path,
                emo_alpha=emo_alpha,
                emo_vector=emo_vec,
                use_random=use_random,
                use_emo_text=use_emo_text,
                emo_text=emo_text,
            )

        finally:
            # 清理我们写到 /tmp 的“材料”文件
            _cleanup_paths(temps_to_cleanup)

    # 返回流并在传输结束后删除生成的 out 文件
    background_tasks.add_task(_remove_file, out_path)
    return StreamingResponse(
        open(out_path, "rb"),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(out_path)}"'},
        background=background_tasks,
    )

# 直接 python 启动（开发）
if __name__ == "__main__":
    import uvicorn
    logger.info("Dev server starting on 0.0.0.0:8000")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
