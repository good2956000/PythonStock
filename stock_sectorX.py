import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import requests
import re
import time
from io import StringIO

# ==========================================
# ★ 修正中文框框問題
# ==========================================
plt.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'PingFang TC', 'Arial Unicode MS', 'SimHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# ★ 1. 動態抓取全台股族群與股票池
# ==========================================
def build_full_sector_map():
    print("正在連線證交所，動態抓取最新上市/上櫃股票清單，請稍候...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    sector_map = {}
    sector_idx = 1
    industry_to_id = {}

    # 2代表上市，4代表上櫃
    targets = [("TW", "2"), ("TWO", "4")]

    for suffix, mode in targets:
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
        try:
            res = requests.get(url, headers=headers, timeout=15)
            res.encoding = 'big5'

            dfs = pd.read_html(StringIO(res.text))
            df = dfs[0]

            df.columns = df.iloc[0]
            df = df.iloc[2:]
            df = df.dropna(subset=['產業別'])

            df['代號'] = df['有價證券代號及名稱'].apply(
                lambda x: re.split(r'[\u3000 ]+', str(x))[0].strip()
            )

            # 只保留 4 碼數字的普通股 (排除特別股、ETF、權證等)
            df = df[df['代號'].str.match(r'^\d{4}$')]

            for _, row in df.iterrows():
                industry = str(row['產業別']).strip()

                if industry not in industry_to_id:
                    industry_to_id[industry] = str(sector_idx)
                    sector_map[str(sector_idx)] = {
                        "name": industry,
                        "tickers": []
                    }
                    sector_idx += 1

                ticker = f"{row['代號']}.{suffix}"
                sector_map[industry_to_id[industry]]["tickers"].append(ticker)

        except Exception as e:
            print(f"抓取 {suffix} 清單時發生錯誤: {e}")

    print(f"成功載入！共計 {len(sector_map)} 個產業族群。\n")
    return sector_map

# 執行函數，自動生成最新的全台股分類字典
SECTOR_MAP = build_full_sector_map()

# ==========================================
# ★ 2. 使用者選擇族群
# ==========================================
print("=" * 50)
print(" 📊 台股族群掃描器 (上市+上櫃)")
print("=" * 50)
print("\n請選擇要掃描的族群：\n")
for key, val in SECTOR_MAP.items():
    count = len(val["tickers"])
    print(f"  [{key:>2}] {val['name']} ({count} 檔)")

print(f"\n  [ 0] 全部掃描")
print()

choice = input("請輸入編號 (可用逗號分隔多選，例如 1,3,5)：").strip()

# 解析使用者選擇
if choice == "0":
    selected_sectors = list(SECTOR_MAP.keys())
else:
    selected_sectors = [s.strip() for s in choice.split(",") if s.strip() in SECTOR_MAP]

if not selected_sectors:
    print("❌ 無效的選擇，程式結束。")
    exit()

# 組合所選族群的股票
tickers = []
sector_names = []
for s in selected_sectors:
    tickers.extend(SECTOR_MAP[s]["tickers"])
    sector_names.append(SECTOR_MAP[s]["name"])

# 去除重複
tickers = list(dict.fromkeys(tickers))
sector_label = "、".join(sector_names)

print(f"\n✅ 已選擇族群：{sector_label}")
print(f"   共計 {len(tickers)} 檔標的，開始掃描...\n")

# ==========================================
# ★ 3. 開始批次處理與條件篩選
# ==========================================
summary_reports = []
strong_watchlist = []      # 爆量強勢：全部 4 個條件
watchlist_no_surge = []   # 候選觀察：條件 1+2+3，無爆量
strong_dfs = {}

for ticker in tickers:
    print(f"正在分析 {ticker} ...")
    time.sleep(1)

    df = yf.download(ticker, period="6mo", progress=False)

    if df.empty:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # 計算均線
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['OSC'] = df['MACD'] - df['Signal']

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI14'] = 100 - (100 / (1 + rs))

    # KD
    low_9 = df['Low'].rolling(window=9, min_periods=1).min()
    high_9 = df['High'].rolling(window=9, min_periods=1).max()
    df['RSV'] = 100 * (df['Close'] - low_9) / (high_9 - low_9)
    df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()

    # 成交量均線
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()

    df = df.dropna()
    if len(df) < 2:
        continue

    # ==========================================
    # ★ 策略條件篩選 (抓底部起漲點版本)
    # ==========================================
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # 1. 均線糾結且剛突破
    ma_max = max(latest['MA5'], latest['MA20'], latest['MA60'])
    ma_min = min(latest['MA5'], latest['MA20'], latest['MA60'])
    is_ma_tangled = ((ma_max - ma_min) / ma_min) < 0.05
    is_breaking_out = latest['Close'] > ma_max

    # 2. KD 低檔黃金交叉 (K<30)
    is_kd_low_golden = (latest['K'] > latest['D']) and (prev['K'] <= prev['D']) and (prev['K'] < 30)

    # 3. MACD 動能翻紅
    is_macd_red = latest['OSC'] > 0

    # 4. 溫和放量 (1.5倍) 與基本流動性 (500張)
    is_liquid = latest['Volume_MA5'] > (500 * 1000)
    is_vol_surge = latest['Volume'] > (latest['Volume_MA5'] * 1.2)

    # ★ 綜合判斷：只要滿足低檔起漲核心條件就選出
    if is_kd_low_golden and is_breaking_out and is_ma_tangled and is_macd_red and is_liquid and is_vol_surge:
        vol_in_lots = latest['Volume'] / 1000
        vol_ma5_lots = latest['Volume_MA5'] / 1000
        strong_watchlist.append(
            f"⭐ {ticker} (收盤: {latest['Close']:.2f}, "
            f"量增: {vol_in_lots:,.0f}張, 底部剛突破!)"
        )
        strong_dfs[ticker] = df
    elif is_kd_low_golden and is_ma_tangled and is_liquid:
        # 均線糾結 + KD 低檔金叉，但尚未放量突破，列為候選觀察
        vol_ma5_lots = latest['Volume_MA5'] / 1000
        watchlist_no_surge.append(
            f"🔔 {ticker} (收盤: {latest['Close']:.2f}, "
            f"均量: {vol_ma5_lots:,.0f}張，均線糾結待突破)"
        )

    report = f"【{ticker}】 收盤: {latest['Close']:.2f} | K: {latest['K']:.2f} | D: {latest['D']:.2f} "
    if latest['K'] > 80:
        report += "-> 🔥 短線過熱"
    elif latest['K'] < latest['D']:
        report += "-> 📉 KD 偏弱"
    else:
        report += "-> 📈 多方優勢"

    summary_reports.append(report)

# ==========================================
# ★ 5. 顯示文字報告
# ==========================================
print("\n" + "=" * 50)
print(f" 🌟 【{sector_label}】爆量強勢名單 (站上月線 + KD金叉 + 爆量) 🌟")
print("=" * 50)
if strong_watchlist:
    for item in strong_watchlist:
        print(item)
    print("\n📊 稍後將為以上強勢標的自動產生技術線型圖表...")
else:
    print("今日無符合全部 4 個條件（含爆量）的股票。")

print("\n" + "=" * 50)
print(f" 🔔 【{sector_label}】候選觀察名單 (站上月線 + KD金叉，待爆量確認)")
print("=" * 50)
if watchlist_no_surge:
    for item in watchlist_no_surge:
        print(item)
else:
    print("今日無符合前 3 個條件的候選股票。")

print("\n" + "=" * 50)
print(" 📋 族群狀態總覽 📋")
print("=" * 50)
for report in summary_reports:
    print(report)

# ==========================================
# ★ 6. 建立文字報告視窗
# ==========================================
fig_text, ax_text = plt.subplots(figsize=(10, max(6, len(summary_reports) * 0.4 + 3)))
ax_text.axis('off')
fig_text.suptitle(f'【{sector_label}】族群掃描報告', fontsize=14, fontweight='bold')

lines = []
lines.append(f'【強勢名單】站上月線 + KD 金叉')
lines.append('=' * 48)
if strong_watchlist:
    for item in strong_watchlist:
        lines.append(item.replace('★ ', '[★] '))
else:
    lines.append('今日無符合條件的股票。')
lines.append('')
lines.append('【族群狀態總覽】')
lines.append('=' * 48)
for item in summary_reports:
    clean_item = (item
                  .replace('-> 🔥 短線過熱', '-> [過熱]')
                  .replace('-> 📉 KD 偏弱',  '-> [偏弱]')
                  .replace('-> 📈 多方優勢',  '-> [多方]'))
    lines.append(clean_item)

report_text = '\n'.join(lines)
ax_text.text(0.02, 0.98, report_text, transform=ax_text.transAxes,
             fontsize=9, verticalalignment='top', wrap=True)
plt.tight_layout()

# ==========================================
# ★ 7. 專屬繪圖：只畫強勢股的圖表
# ==========================================
if strong_dfs:
    print("\n正在生成強勢股圖表...")
    for tkr, data in strong_dfs.items():
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, figsize=(12, 13), gridspec_kw={'height_ratios': [3, 1, 1, 1]}
        )
        fig.suptitle(f'強勢突破：{tkr} 技術分析', fontsize=16, color='darkred', fontweight='bold')

        # 均線子圖
        ax1.plot(data.index, data['Close'], label='收盤價', color='black', linewidth=1.5)
        ax1.plot(data.index, data['MA5'], label='5MA', color='blue', alpha=0.7)
        ax1.plot(data.index, data['MA20'], label='20MA (月線)', color='magenta', linewidth=2)
        ax1.plot(data.index, data['MA60'], label='60MA (季線)', color='green', alpha=0.7)
        ax1.set_ylabel('股價')
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # MACD 子圖
        ax2.plot(data.index, data['MACD'], label='MACD', color='blue')
        ax2.plot(data.index, data['Signal'], label='Signal', color='orange')
        colors = ['red' if val >= 0 else 'green' for val in data['OSC']]
        ax2.bar(data.index, data['OSC'], color=colors, alpha=0.5)
        ax2.axhline(0, color='black', linewidth=1)
        ax2.set_ylabel('MACD')
        ax2.legend(loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.5)

        # RSI 子圖
        ax3.plot(data.index, data['RSI14'], label='RSI (14)', color='purple')
        ax3.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax3.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax3.set_ylabel('RSI')
        ax3.legend(loc='upper left')
        ax3.grid(True, linestyle='--', alpha=0.5)

        # KD 子圖
        ax4.plot(data.index, data['K'], label='K (9)', color='blue')
        ax4.plot(data.index, data['D'], label='D (9)', color='orange')
        ax4.axhline(80, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(20, color='green', linestyle='--', alpha=0.5)
        ax4.set_ylabel('KD')
        ax4.legend(loc='upper left')
        ax4.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

# ★ 將原本最後一行的 plt.show() 刪除或加上 # 註解掉
    plt.show()
    # print("\n所有圖表皆已產生並存檔完畢！")
