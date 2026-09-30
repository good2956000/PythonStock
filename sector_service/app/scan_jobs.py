"""族群掃描工作管理（Application 層）。

全市場掃描需要數分鐘，無法在單一 HTTP 請求內完成，
因此改為「建立工作 → 背景執行 → 前端輪詢進度」的模式。
"""
import logging
import threading
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pandas as pd

from .indicators import STRATEGY_STANDARD, add_indicators, evaluate_signals

logger = logging.getLogger(__name__)

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# 下載函式簽章：接收一批 ticker，回傳 {ticker: 日 K DataFrame}
PriceDownloader = Callable[[List[str]], Dict[str, pd.DataFrame]]


@dataclass
class StockScanResult:
    ticker: str
    trade_date: str
    close: float
    ma20: float
    k: float
    d: float
    volume_lots: float
    volume_ma5_lots: float
    category: Optional[str]
    trend: str
    conditions: Dict[str, bool]


@dataclass
class ScanJob:
    job_id: str
    sector_names: List[str]
    tickers: List[str]
    strategy: str = STRATEGY_STANDARD
    status: str = STATUS_QUEUED
    processed_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: Optional[str] = None
    error_message: Optional[str] = None
    results: List[StockScanResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["total_count"] = len(self.tickers)
        del data["tickers"]
        return data


class ScanJobManager:
    """以單一背景執行緒依序處理掃描工作，避免同時對 Yahoo 發出大量請求。"""

    def __init__(self, price_downloader: PriceDownloader, batch_size: int = 50, max_jobs_kept: int = 20):
        self._price_downloader = price_downloader
        self._batch_size = batch_size
        self._max_jobs_kept = max_jobs_kept
        self._jobs: "OrderedDict[str, ScanJob]" = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sector-scan")

    def submit(self, sector_names: List[str], tickers: List[str], strategy: str = STRATEGY_STANDARD) -> ScanJob:
        # 去除重複並保留原順序
        unique_tickers = list(dict.fromkeys(tickers))
        job = ScanJob(job_id=uuid.uuid4().hex, sector_names=sector_names, tickers=unique_tickers, strategy=strategy)
        with self._lock:
            self._jobs[job.job_id] = job
            # 只保留最近的工作，避免記憶體持續成長
            while len(self._jobs) > self._max_jobs_kept:
                self._jobs.popitem(last=False)
        self._executor.submit(self._run, job)
        return job

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def run_synchronously(
        self, sector_names: List[str], tickers: List[str], strategy: str = STRATEGY_STANDARD
    ) -> ScanJob:
        """測試用：不經背景執行緒直接執行。"""
        job = ScanJob(
            job_id=uuid.uuid4().hex, sector_names=sector_names, tickers=list(dict.fromkeys(tickers)), strategy=strategy
        )
        self._run(job)
        return job

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _run(self, job: ScanJob) -> None:
        self._update(job, status=STATUS_RUNNING)
        try:
            for start in range(0, len(job.tickers), self._batch_size):
                batch = job.tickers[start:start + self._batch_size]
                price_frames = self._price_downloader(batch)
                batch_results = []
                for ticker in batch:
                    frame = price_frames.get(ticker)
                    result = self._analyze(ticker, frame, job.strategy) if frame is not None else None
                    if result is not None:
                        batch_results.append(result)
                with self._lock:
                    job.results.extend(batch_results)
                    job.processed_count += len(batch)
            self._update(job, status=STATUS_COMPLETED)
        except Exception as error:  # 背景執行緒的例外必須記錄下來，否則前端只會一直等待
            logger.exception("掃描工作 %s 失敗", job.job_id)
            self._update(job, status=STATUS_FAILED, error_message=str(error))

    def _update(self, job: ScanJob, status: str, error_message: Optional[str] = None) -> None:
        with self._lock:
            job.status = status
            if error_message is not None:
                job.error_message = error_message
            if status in (STATUS_COMPLETED, STATUS_FAILED):
                job.finished_at = datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _analyze(ticker: str, price_frame: pd.DataFrame, strategy: str) -> Optional[StockScanResult]:
        signal = evaluate_signals(add_indicators(price_frame), strategy)
        if signal is None:
            return None
        return StockScanResult(
            ticker=ticker,
            trade_date=signal.trade_date,
            close=round(signal.close, 2),
            ma20=round(signal.ma20, 2),
            k=round(signal.k, 2),
            d=round(signal.d, 2),
            volume_lots=round(signal.volume / 1000),
            volume_ma5_lots=round(signal.volume_ma5 / 1000),
            category=signal.category,
            trend=signal.trend,
            conditions=signal.conditions,
        )
