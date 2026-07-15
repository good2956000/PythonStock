import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# ★ 修正中文框框問題
# ==========================================
plt.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'PingFang TC', 'Arial Unicode MS', 'SimHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# ★ 1. 族群股票池定義 (上市 .TW / 上櫃 .TWO)
# ==========================================
SECTOR_MAP = {
    "1": {
        "name": "半導體",
        "tickers": [
            "2330.TW", "2303.TW", "2454.TW", "3711.TW", "2379.TW",
            "3034.TW", "6488.TW", "2344.TW", "3529.TW", "6770.TW",
            "5274.TWO", "3661.TW", "8150.TWO", "6147.TWO", "3443.TW",
            "2408.TW", "3105.TW", "2436.TW", "6239.TW", "3037.TW",
        ]
    },
    "2": {
        "name": "電子零組件",
        "tickers": [
            "2327.TW", "3017.TW", "2308.TW", "3023.TW", "2059.TW",
            "3533.TW", "2368.TW", "6285.TW", "3036.TW", "2345.TW",
            "3665.TWO", "6164.TWO", "4956.TWO", "3038.TWO", "8928.TWO",
        ]
    },
    "3": {
        "name": "金融保險",
        "tickers": [
            "2881.TW", "2882.TW", "2883.TW", "2884.TW", "2885.TW",
            "2886.TW", "2887.TW", "2888.TW", "2889.TW", "2890.TW",
            "2891.TW", "2892.TW", "5880.TW", "2801.TW", "2812.TW",
            "2816.TW", "2834.TW", "2838.TW", "2845.TW", "2867.TW",
        ]
    },
    "4": {
        "name": "營建建材",
        "tickers": [
            "2501.TW", "2504.TW", "2505.TW", "2511.TW", "2515.TW",
            "2520.TW", "2524.TW", "2527.TW", "2528.TW", "2534.TW",
            "2535.TW", "2538.TW", "2539.TW", "2542.TW", "2543.TW",
            "2545.TW", "2546.TW", "2547.TW", "2548.TW", "3703.TW",
            "5515.TW", "5519.TW", "5522.TW", "5534.TW",
            "5536.TWO", "2597.TWO", "5531.TWO",
        ]
    },
    "5": {
        "name": "航運",
        "tickers": [
            "2603.TW", "2609.TW", "2615.TW", "2605.TW", "2606.TW",
            "2607.TW", "2608.TW", "2610.TW", "2611.TW", "2612.TW",
            "2613.TW", "2614.TW", "2616.TW", "2617.TW", "2618.TW",
            "5765.TWO",
        ]
    },
    "6": {
        "name": "鋼鐵",
        "tickers": [
            "2002.TW", "2006.TW", "2008.TW", "2010.TW", "2012.TW",
            "2013.TW", "2014.TW", "2015.TW", "2020.TW", "2022.TW",
            "2023.TW", "2024.TW", "2025.TW", "2027.TW", "2029.TW",
            "2032.TW", "2034.TW", "5765.TWO",
        ]
    },
    "7": {
        "name": "生技醫療",
        "tickers": [
            "1707.TW", "4904.TW", "1701.TW", "1702.TW", "1720.TW",
            "6446.TW", "4743.TW", "1760.TW", "4746.TW", "4142.TWO",
            "6472.TWO", "4147.TWO", "4174.TWO", "4126.TWO", "6789.TWO",
            "4168.TWO", "6547.TWO", "4113.TWO", "4192.TWO", "6869.TWO",
        ]
    },
    "8": {
        "name": "電腦及週邊",
        "tickers": [
            "2353.TW", "2356.TW", "2357.TW", "2382.TW", "3231.TW",
            "2324.TW", "3013.TW", "6669.TW", "3706.TW", "2365.TW",
            "3005.TW", "2395.TW", "3044.TWO", "3260.TWO", "6411.TWO",
        ]
    },
    "9": {
        "name": "光電",
        "tickers": [
            "2409.TW", "3481.TW", "2393.TW", "6176.TW", "3406.TW",
            "6116.TW", "2426.TW", "6271.TW", "3019.TW", "6245.TWO",
            "3630.TWO", "4966.TWO", "6244.TWO", "3313.TWO", "4961.TWO",
        ]
    },
    "10": {
        "name": "通信網路",
        "tickers": [
            "2412.TW", "3045.TW", "4904.TW", "6285.TW", "3596.TW",
            "2332.TW", "3682.TW", "4906.TW", "2439.TW", "6218.TW",
            "3163.TWO", "6285.TWO", "5765.TWO",
        ]
    },
    "11": {
        "name": "食品",
        "tickers": [
            "1216.TW", "1301.TW", "2912.TW", "1215.TW", "1227.TW",
            "1229.TW", "1231.TW", "1232.TW", "1233.TW", "1234.TW",
            "1235.TW", "1236.TW", "1264.TW", "4205.TWO", "1258.TWO",
        ]
    },
    "12": {
        "name": "觀光餐飲",
        "tickers": [
            "2702.TW", "2704.TW", "2705.TW", "2706.TW", "2707.TW",
            "2712.TW", "2719.TW", "2722.TW", "2723.TW", "2727.TW",
            "2729.TW", "2731.TW", "2739.TW", "2740.TWO", "2753.TWO",
        ]
    },
    "13": {
        "name": "汽車",
        "tickers": [
            "2201.TW", "2204.TW", "2206.TW", "2207.TW", "2208.TW",
            "2227.TW", "2231.TW", "2233.TW", "2236.TW", "6625.TW",
            "2250.TWO",
        ]
    },
    "14": {
        "name": "AI伺服器/散熱",
        "tickers": [
            "2317.TW", "3231.TW", "2382.TW", "6669.TW", "3706.TW",
            "2345.TW", "3017.TW", "6515.TW", "3005.TW", "2395.TW",
            "3059.TWO", "6890.TWO", "3260.TWO", "8942.TWO", "3178.TWO",
        ]
    },
}

# ==========================================
# ★ 2. 使用者選擇族群
# ==========================================
print("=" * 50)
print(" ?? 台股族群掃描器 (上市+上櫃)")
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
    print("? 無效的選擇，程式結束。")
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

print(f"\n? 已選擇族群：{sector_label}")
print(f"   共計 {len(tickers)} 檔標的，開始掃描...\n")

# ==========================================
# ★ 3. 開始批次處理與條件篩選
# ==========================================
summary_reports = []
strong_watchlist = []
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

    df = df.dropna()
    if len(df) < 2:
        continue

    # ==========================================
    # ★ 4. 策略條件篩選
    # ==========================================
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    is_above_ma20 = latest['Close'] > latest['MA20']
    is_kd_golden = (latest['K'] > latest['D']) and (prev['K'] <= prev['D'])

    if is_above_ma20 and is_kd_golden:
        strong_watchlist.append(
            f"? {ticker} (收盤價: {latest['Close']:.2f}, 月線: {latest['MA20']:.2f}, K值: {latest['K']:.2f})"
        )
        strong_dfs[ticker] = df

    report = f"【{ticker}】 收盤: {latest['Close']:.2f} | K: {latest['K']:.2f} | D: {latest['D']:.2f} "
    if latest['K'] > 80:
        report += "-> ?? 短線過熱"
    elif latest['K'] < latest['D']:
        report += "-> ?? KD 偏弱"
    else:
        report += "-> ? 多方優勢"

    summary_reports.append(report)

# ==========================================
# ★ 5. 顯示文字報告
# ==========================================
print("\n" + "=" * 50)
print(f" ?? 【{sector_label}】強勢觀察名單 (站上月線 + KD金叉) ??")
print("=" * 50)
if len(strong_watchlist) > 0:
    for item in strong_watchlist:
        print(item)
    print("\n?? 稍後將為以上強勢標的自動產生技術線型圖表...")
else:
    print("今日無符合「站上月線且KD黃金交叉」條件的股票。")

print("\n" + "=" * 50)
print(" ?? 族群狀態總覽 ??")
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
        lines.append(item.replace('? ', '[*] '))
else:
    lines.append('今日無符合條件的股票。')
lines.append('')
lines.append('【族群狀態總覽】')
lines.append('=' * 48)
for item in summary_reports:
    clean_item = (item
                  .replace('-> ?? 短線過熱', '-> [過熱]')
                  .replace('-> ?? KD 偏弱',  '-> [偏弱]')
                  .replace('-> ? 多方優勢',  '-> [多方]'))
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

plt.show()
