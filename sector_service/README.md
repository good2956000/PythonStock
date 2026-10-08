# Python 選股策略服務（FastAPI）

把 `PythonStock/` 資料夾內的腳本改寫成 API，提供給 ASP.NET 網站「Python選股」選單下的頁面使用。

## 腳本與頁面對照

| 原腳本 | 網站頁面 | API |
|---|---|---|
| `stock_sector.py` | 族群掃描：爆量強勢 | `POST /api/scans`（`strategy: standard`） |
| `stock_sectorX.py` | 族群掃描：底部起漲 | `POST /api/scans`（`strategy: bottom_breakout`） |
| `stock.py` | 族群掃描：營建股強勢（固定清單 `P1`） | `POST /api/scans`（`strategy: basic`） |
| `stock_bias_bollinger.py` | 個股 RSI／MACD／KD 訊號 | `GET /api/stocks/{code}/signal-analysis` |
| `stock_all.py` | 大盤 KD 波段回測 0050 | `GET /api/market/kd-backtest` |
| `market_rebound.py` | 回檔後隔月反彈統計 | `GET /api/market/rebound` |
| `market_margin.py` | 融資餘額與維持率 | `GET /api/market/margin` |

原腳本保留不動，仍可在命令列單獨執行。

## 架構

| 檔案 | 分層 | 說明 |
|---|---|---|
| `app/indicators.py` | Domain | 族群掃描的指標計算與三種選股策略（純函式） |
| `app/signal_analysis.py` | Domain | 個股訊號、右側順勢進場計分與動態出場回測 |
| `app/market_analysis.py` | Domain | 大盤 KD 波段回測、回檔反彈統計、融資維持率估算 |
| `app/serialization.py` | Domain | pandas / numpy 結果轉 JSON（NaN → null） |
| `app/market_data.py` | Infrastructure | 證交所產業清單與融資資料、Yahoo Finance 日 K 下載與快取 |
| `app/scan_jobs.py` | Application | 背景掃描工作（單一執行緒依序處理） |
| `app/main.py` | Presentation | FastAPI 端點、輸入驗證與 API Key 驗證 |

## 安裝與啟動（開發環境）

```powershell
cd PythonStock\sector_service
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 產生一組隨機金鑰（兩邊需設定相同的值，不要寫進任何設定檔）
$env:SECTOR_API_KEY = .venv\Scripts\python -c "import secrets; print(secrets.token_hex(32))"
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

ASP.NET 網站端以環境變數帶入同一把金鑰後再啟動：

```powershell
$env:SectorScannerApi__ApiKey = "<與 SECTOR_API_KEY 相同>"
dotnet run
```

部署到本機 IIS 請改用 `deploy/Deploy-SectorScan.ps1`（需系統管理員權限）。

服務網址設定於 `appsettings.json` 的 `SectorScannerApi:BaseUrl`（預設 `http://127.0.0.1:8001`）。
服務只綁定 `127.0.0.1`，不直接對外開放；瀏覽器一律經由 ASP.NET 的 `SectorScanController` / `PythonStrategyController` 轉呼叫。

## API

所有 `/api/*` 端點都需要 `X-Api-Key` 標頭。互動式文件：服務啟動後開啟 `http://127.0.0.1:8001/docs`。

| 方法 | 路徑 | 參數 | 說明 |
|---|---|---|---|
| GET | `/health` | | 健康檢查（不需金鑰） |
| GET | `/api/strategies` | | 族群掃描策略清單 |
| GET | `/api/sectors` | | 固定清單＋產業族群（快取 12 小時，`SECTOR_CACHE_SECONDS`） |
| POST | `/api/scans` | `{"sectorIds": ["P1","9"], "strategy": "standard"}` | 建立掃描工作；空陣列或含 `"0"` 代表全部產業族群 |
| GET | `/api/scans/{jobId}` | | 查詢進度與結果 |
| GET | `/api/stocks/{ticker}/indicators` | | 單一股票指標序列（`2330.TW`、`6488.TWO`） |
| GET | `/api/stocks/{code}/signal-analysis` | `period`=3mo/6mo/1y/2y/5y | 純數字代號先試上市、查無再試上櫃 |
| GET | `/api/market/kd-backtest` | `startDate`=YYYY-MM-DD（預設 2020-06-01） | |
| GET | `/api/market/rebound` | `period`=2y/5y/10y/max（預設 5y） | |
| GET | `/api/market/margin` | `days`=10~120（預設 60） | 首次約 30~40 秒，已公布日期會快取 |

## 快取與限流

- 族群清單（證交所 ISIN 頁約 9 MB，回應速度波動大，曾超過 30 秒）：
  - 快取過期時先回傳舊清單，同時在背景更新，使用者不需等待
  - 每次更新成功都寫入 `cache/sectors.json`（可用 `SECTOR_CACHE_FILE` 指定），服務重新啟動後立即可用
  - 服務啟動時即在背景預先載入；單次下載總時間上限 90 秒；上市或上櫃任一失敗時保留舊清單，不會存入不完整的資料
- Yahoo Finance 會對短時間大量請求回傳 `YFRateLimitError`，因此單一標的下載結果快取 10 分鐘
- 族群批次下載每批 50 檔，批次間延遲 `DOWNLOAD_CHUNK_DELAY_SECONDS`（預設 1 秒）
- 證交所融資資料逐日請求間隔 0.3 秒，成功取得的日期永久快取（服務重啟後清空）

## 測試

```powershell
.venv\Scripts\python -m pytest -q
```

## 與原腳本的差異

- 移除 `input()` 互動與 matplotlib / Plotly 視窗，改由網頁輸入參數並以 Chart.js / Plotly.js 繪圖
- 族群掃描股價改為每批 50 檔一次下載，不再每檔等 1 秒
- 族群掃描：最近 9 日最高價等於最低價時 RSV 以 50 代替，避免最新交易日因 NaN 被刪除
- 爆量倍數沿用程式碼實際值 1.2（`stock_sector.py` 註解寫 2 倍、`stock_sectorX.py` 註解寫 1.5 倍），定義於 `indicators.VOLUME_SURGE_RATIO`
- 底部起漲（`stock_sectorX.py`）新增第 7 個條件「月線大於季線（MA20 > MA60）」，7 個條件全部符合才列為「底部剛突破」；
  原本的「均線糾結待突破」觀察名單改為網頁上的「自選條件」，可任意勾選 7 個條件中的幾項即時篩選
  （預設勾選 ⑤流動性 ⑦月線大於季線 ⑧月線乖離率 < 6%），條件定義見 `indicators.BOTTOM_BREAKOUT_CONDITIONS`
- `market_margin.py` 檔案內接了兩個版本，採用前半段「每日 MI_MARGN」版本；後半段「每月 API」目前證交所回傳格式已無 `data` 欄位，原腳本執行時會失敗
