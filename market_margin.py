"""
大盤融資餘額 + 融券餘額 + 加權指數 趨勢追蹤器
資料來源：
  - 加權指數  : yfinance (^TWII)
  - 融資/融券餘額 : 證交所 TWSE 公開 API (MI_MARGN)
備註：融資維持率需 擔保品市值/融資金額，證交所未直接提供，
      本程式以「融資餘額（億元）」作為市場槓桿程度的代理指標。
"""
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import timedelta
import time

# ==========================================
# ★ 修正中文顯示
# ==========================================
plt.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'PingFang TC', 'Arial Unicode MS', 'SimHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# ★ 1. 參數設定
# ==========================================
RECENT_DAYS  = 60       # 抓最近幾個交易日（建議 30~120）
INDEX_TICKER = "^TWII"
DELAY        = 0.3      # API 請求間隔（秒）

# ==========================================
# ★ 2. 下載加權指數，取出最近 N 個交易日期
# ==========================================
print(f"📥 正在下載加權指數 ({INDEX_TICKER})...")
taiex = yf.download(INDEX_TICKER, period="6mo", progress=False)
if isinstance(taiex.columns, pd.MultiIndex):
    taiex.columns = taiex.columns.droplevel(1)
taiex = taiex[['Close']].copy()
taiex.index = pd.to_datetime(taiex.index)

trade_dates = taiex.index[-RECENT_DAYS:]   # 最近 N 個交易日
print(f"✅ 加權指數：{len(taiex)} 筆，取最近 {len(trade_dates)} 筆做融資比對")

# ==========================================
# ★ 3. 逐日抓取 TWSE 融資融券資料
# ==========================================
API_URL = "https://www.twse.com.tw/exchangeReport/MI_MARGN"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def fetch_margin_day(dt):
    """抓取單一交易日的整體融資融券資料，回傳 dict 或 None"""
    date_str = dt.strftime('%Y%m%d')
    params   = {'date': date_str, 'selectType': 'MS', 'response': 'json'}
    try:
        res  = requests.get(API_URL, params=params, headers=HEADERS, timeout=10)
        data = res.json()
        if data.get('stat') != 'OK':
            return None
        rows = data['tables'][0]['data']   # list of rows
        # 解析欄位：今日餘額 = index 5
        result = {'date': dt}
        for row in rows:
            name = row[0]
            today_val = row[5].replace(',', '')
            if '融資(交易單位)' in name:
                result['融資餘額(張)'] = int(today_val)
            elif '融券(交易單位)' in name:
                result['融券餘額(張)'] = int(today_val)
            elif '融資金額' in name:
                result['融資餘額(仟元)'] = int(today_val)
        return result
    except Exception:
        return None

print(f"\n📥 逐日下載融資資料（{len(trade_dates)} 日）...")
records = []
for i, dt in enumerate(trade_dates):
    rec = fetch_margin_day(dt)
    if rec:
        records.append(rec)
        if (i + 1) % 10 == 0:
            print(f"  ✅ 已完成 {i+1}/{len(trade_dates)} 日")
    time.sleep(DELAY)

if not records:
    print("❌ 融資資料下載失敗，請確認網路連線。")
    exit()

margin_df = pd.DataFrame(records).set_index('date')
margin_df.index = pd.to_datetime(margin_df.index)
print(f"\n✅ 成功取得 {len(margin_df)} 筆融資資料")

# ==========================================
# ★ 4. 合併加權指數與融資資料
# ==========================================
combined = taiex.join(margin_df, how='inner').dropna()
combined['融資餘額(億元)'] = combined['融資餘額(仟元)'] / 100_000
combined['融券餘額(萬張)'] = combined['融券餘額(張)']  / 10_000

# ==========================================
# ★ 計算近似融資維持率
# ==========================================
# 原理：維持率 = 擔保品市值 / 融資金額
# 擔保品市值 ≈ 融資張數 × 平均建倉股價 × (現在指數/建倉指數)
# 建倉指數代理：TAIEX 60日移動平均 (約含近 3 個月建倉)
# 融資成數假設：0.60 (台股多數股票適用)
MARGIN_RATE = 0.60

# 需要更長的 TAIEX 歷史計算 60 日均線
taiex_full = yf.download(INDEX_TICKER, period='6mo', progress=False)
if isinstance(taiex_full.columns, pd.MultiIndex):
    taiex_full.columns = taiex_full.columns.droplevel(1)
taiex_full.index = pd.to_datetime(taiex_full.index)
taiex_ma60 = taiex_full['Close'].rolling(60).mean()

# 對齊 combined 的日期
ma60_aligned = taiex_ma60.reindex(combined.index)
combined['TAIEX_MA60'] = ma60_aligned

# 近似維持率 = TAIEX_現在 / (融資成數 × TAIEX_60日均)
combined['近似維持率(%)'] = (
    combined['Close'] / (MARGIN_RATE * combined['TAIEX_MA60']) * 100
).round(1)
combined = combined.dropna(subset=['近似維持率(%)'])

print(f"\n【注意】融資維持率說明：TWSE 公開 API 未提供擔保品市值，")
print(f"    本程式以 [TAIEX ÷ (融資成數0.6 × 60日均指數)] 作為近似估算，")
print(f"    僅供趨勢參考，非精確數值。")

# ==========================================
# ★ 5. 印出近 10 日報表
# ==========================================
print("\n" + "=" * 75)
print(" 📋 近 10 個交易日 — 加權指數 / 融資餘額 / 融券餘額 / 近似維持率")
print("=" * 75)
show = combined[['Close', '融資餘額(億元)', '融券餘額(萬張)', '近似維持率(%)']].tail(10).copy()
show.columns = ['加權指數', '融資餘額(億元)', '融券餘額(萬張)', '近似維持率(%)']
show.index = show.index.strftime('%Y-%m-%d')
show['加權指數']      = show['加權指數'].round(0).astype(int)
show['融資餘額(億元)'] = show['融資餘額(億元)'].round(0).astype(int)
show['融券餘額(萬張)'] = show['融券餘額(萬張)'].round(1)
print(show.to_string())
print("\n  * 近似維持率 < 140% 請留意，< 130% 為融資追繳警戒區")

# ==========================================
# ★ 6. 繪圖（三軸）
# ==========================================
fig, (ax1, ax_rate) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
fig.suptitle(
    f'台股加權指數 / 融資餘額 / 近似融資維持率（近 {len(combined)} 個交易日）\n'
    f'【注意】維持率為估算值（公式：TAIEX ÷ 融資成數0.6 ÷ TAIEX_60MA × 100）',
    fontsize=13, fontweight='bold'
)

x = combined.index

# ==================== 上圖：加權指數 + 融資餘額 ====================
color_idx  = 'steelblue'
color_loan = 'tomato'
color_short= 'seagreen'

ax1.set_ylabel('加權指數（點）', color=color_idx)
ax1.plot(x, combined['Close'], color=color_idx, linewidth=2, label='加權指數')
ax1.tick_params(axis='y', labelcolor=color_idx)
ax1.grid(axis='x', linestyle='--', alpha=0.3)

ax2 = ax1.twinx()
ax2.set_ylabel('融資餘額（億元）', color=color_loan)
ax2.fill_between(x, combined['融資餘額(億元)'], alpha=0.2, color=color_loan)
ax2.plot(x, combined['融資餘額(億元)'], color=color_loan, linewidth=1.5,
         linestyle='--', label='融資餘額')
ax2.tick_params(axis='y', labelcolor=color_loan)

lines  = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax1.legend(lines, labels, loc='upper left')

# ==================== 下圖：近似融資維持率 ====================
rate = combined['近似維持率(%)']
ax_rate.set_ylabel('近似融資維持率 (%)')
ax_rate.set_xlabel('日期')
ax_rate.plot(x, rate, color='darkorange', linewidth=2, label='近似維持率')
ax_rate.fill_between(x, rate, 130, where=(rate < 160), alpha=0.15, color='orange')
ax_rate.axhline(130, color='red',    linestyle='--', linewidth=1.5, label='130% 追繳警戒')
ax_rate.axhline(150, color='orange', linestyle=':', linewidth=1,   label='150% 注意線')
ax_rate.axhline(160, color='green',  linestyle=':', linewidth=1,   label='160% 安全線')
ax_rate.legend(loc='upper left')
ax_rate.grid(axis='both', linestyle='--', alpha=0.3)
ax_rate.set_ylim(max(100, rate.min() - 10), rate.max() + 10)

# 標注最新維持率
last_rate = rate.iloc[-1]
ax_rate.annotate(f'現在: {last_rate:.1f}%',
                 xy=(x[-1], last_rate),
                 xytext=(-60, 8), textcoords='offset points',
                 fontsize=10, fontweight='bold',
                 color='red' if last_rate < 140 else 'darkorange')

ax_rate.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax_rate.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
print("\n✅ 完成！")

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from datetime import date, timedelta
import time

# ==========================================
# ★ 修正中文顯示
# ==========================================
plt.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'PingFang TC', 'Arial Unicode MS', 'SimHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# ★ 1. 參數設定
# ==========================================
MONTHS_BACK  = 6        # 往回抓幾個月
INDEX_TICKER = "^TWII"  # 加權指數

# ==========================================
# ★ 2. 下載加權指數
# ==========================================
print(f"📥 正在下載加權指數 ({INDEX_TICKER})...")
taiex = yf.download(INDEX_TICKER, period=f"{MONTHS_BACK}mo", progress=False)
if isinstance(taiex.columns, pd.MultiIndex):
    taiex.columns = taiex.columns.droplevel(1)
taiex = taiex[['Close']].copy()
taiex.index = pd.to_datetime(taiex.index)
print(f"✅ 加權指數：{len(taiex)} 筆（{taiex.index[0].date()} ~ {taiex.index[-1].date()}）")

# ==========================================
# ★ 3. 從 TWSE 抓取融資餘額（每月一次 API）
# ==========================================
def roc_to_date(roc_str):
    """民國日期 '113/07/15' → datetime"""
    try:
        parts = roc_str.strip().split('/')
        year  = int(parts[0]) + 1911
        return pd.Timestamp(f"{year}/{parts[1]}/{parts[2]}")
    except Exception:
        return pd.NaT

def fetch_margin_month(year, month):
    """抓取指定月份的整體市場融資融券資料"""
    date_str = f"{year}{month:02d}01"
    url    = "https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
    params = {'date': date_str, 'selectType': 'MS', 'response': 'json'}
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        res  = requests.get(url, params=params, headers=headers, timeout=12)
        data = res.json()
        if data.get('stat') == 'OK' and data.get('data'):
            df = pd.DataFrame(data['data'], columns=data['fields'])
            return df
    except Exception as e:
        print(f"  ⚠️ {year}/{month:02d} 抓取失敗: {e}")
    return None

print(f"\n📥 正在下載近 {MONTHS_BACK} 個月融資餘額...")
frames = []
today = date.today()
for i in range(MONTHS_BACK, -1, -1):
    d = (today.replace(day=1) - timedelta(days=30 * i))
    df_m = fetch_margin_month(d.year, d.month)
    if df_m is not None:
        frames.append(df_m)
        print(f"  ✅ {d.year}/{d.month:02d} — {len(df_m)} 筆")
    time.sleep(0.4)

if not frames:
    print("❌ 融資資料下載失敗，請確認網路連線。")
    exit()

margin_raw = pd.concat(frames, ignore_index=True)

# ==========================================
# ★ 4. 解析欄位：找出日期、融資餘額、融券餘額
# ==========================================
print(f"\n📋 欄位：{list(margin_raw.columns)}\n")

# 自動偵測日期欄
date_col = margin_raw.columns[0]
margin_raw['Date'] = margin_raw[date_col].apply(roc_to_date)
margin_raw = margin_raw.dropna(subset=['Date'])
margin_raw = margin_raw.sort_values('Date').reset_index(drop=True)

# 自動偵測融資(金額)_餘額 欄
loan_col  = [c for c in margin_raw.columns if '融資' in c and '餘額' in c and '金額' in c]
short_col = [c for c in margin_raw.columns if '融券' in c and '餘額' in c and '單位' in c]

if not loan_col:
    # 備用：找任何含「融資」和「餘額」的欄
    loan_col = [c for c in margin_raw.columns if '融資' in c and '餘額' in c]

print(f"融資餘額欄：{loan_col}")
print(f"融券餘額欄：{short_col}")

def clean_num(s):
    try:
        return float(str(s).replace(',', '').replace('--', '').strip())
    except Exception:
        return None

if loan_col:
    margin_raw['融資餘額(千元)'] = margin_raw[loan_col[0]].apply(clean_num)
if short_col:
    margin_raw['融券餘額(千股)'] = margin_raw[short_col[0]].apply(clean_num)

margin_df = margin_raw.set_index('Date')

# ==========================================
# ★ 5. 對齊指數與融資資料（取交集日期）
# ==========================================
combined = taiex.join(margin_df[['融資餘額(千元)'] + (['融券餘額(千股)'] if short_col else [])],
                      how='inner').dropna()

print(f"\n✅ 對齊後共 {len(combined)} 筆交易日資料")
print(combined.tail(5).to_string())

# ==========================================
# ★ 6. 繪圖：雙軸（加權指數 + 融資餘額）
# ==========================================
fig, ax1 = plt.subplots(figsize=(14, 6))
fig.suptitle(f'台股加權指數 vs 市場整體融資餘額（近 {MONTHS_BACK} 個月）',
             fontsize=14, fontweight='bold')

# 左軸：加權指數
color_idx = 'steelblue'
ax1.set_xlabel('日期')
ax1.set_ylabel('加權指數（點）', color=color_idx)
ax1.plot(combined.index, combined['Close'], color=color_idx, linewidth=1.8, label='加權指數')
ax1.tick_params(axis='y', labelcolor=color_idx)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
plt.xticks(rotation=45)
ax1.grid(axis='x', linestyle='--', alpha=0.4)

# 右軸：融資餘額
if '融資餘額(千元)' in combined.columns:
    ax2 = ax1.twinx()
    color_margin = 'tomato'
    ax2.set_ylabel('融資餘額（億元）', color=color_margin)
    margin_yi = combined['融資餘額(千元)'] / 1e5   # 千元 → 億元
    ax2.plot(combined.index, margin_yi, color=color_margin,
             linewidth=1.5, linestyle='--', label='融資餘額')
    ax2.tick_params(axis='y', labelcolor=color_margin)

    # 合併圖例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # 標註最新值
    last_date  = combined.index[-1]
    last_idx   = combined['Close'].iloc[-1]
    last_margin = margin_yi.iloc[-1]
    ax1.annotate(f'{last_idx:,.0f}', xy=(last_date, last_idx),
                 xytext=(10, 5), textcoords='offset points',
                 fontsize=9, color=color_idx, fontweight='bold')
    ax2.annotate(f'{last_margin:,.0f} 億', xy=(last_date, last_margin),
                 xytext=(10, -15), textcoords='offset points',
                 fontsize=9, color=color_margin, fontweight='bold')

plt.tight_layout()

# ==========================================
# ★ 7. 印出近 10 日彙整表
# ==========================================
print("\n" + "=" * 55)
print(" 📋 近 10 個交易日 — 加權指數 vs 融資餘額")
print("=" * 55)
display = combined[['Close', '融資餘額(千元)']].tail(10).copy()
display['融資餘額(億元)'] = (display['融資餘額(千元)'] / 1e5).round(1)
display['加權指數'] = display['Close'].round(0).astype(int)
display.index = display.index.strftime('%Y-%m-%d')
print(display[['加權指數', '融資餘額(億元)']].to_string())

plt.show()
print("\n✅ 完成！")
