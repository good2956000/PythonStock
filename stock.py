import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import time  # 新增：用來控制下載速度，避免被 Yahoo 封鎖

# ==========================================
# ★ 修正中文框框問題
# ==========================================
plt.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'PingFang TC', 'Arial Unicode MS', 'SimHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False 

# ==========================================
# ★ 1. 設定營建族群股票池 (上市/上櫃建材營造)
# ==========================================
# 這裡為你整理了台股主要的營建股代號，你可以隨時增刪
tickers = [
    "2501.TW", "2504.TW", "2505.TW", "2511.TW", "2515.TW", 
    "2520.TW", "2524.TW", "2527.TW", "2528.TW", "2534.TW", 
    "2535.TW", "2538.TW", "2539.TW", "2542.TW", "2543.TW", # 皇昌
    "2545.TW", "2546.TW", "2547.TW", "2548.TW", "3703.TW", # 華固, 欣陸
    "5515.TW", "5519.TW", "5522.TW", "5534.TW" 
    # 註: 若有上櫃股票(如 5536 聖暉)，在 yfinance 中字尾要改成 .TWO (例如 "5536.TWO")
]

summary_reports = []
strong_watchlist = []
strong_dfs = {} # 用來儲存強勢股的 DataFrame，稍後畫圖用

print(f"啟動營建族群掃描，共計 {len(tickers)} 檔標的...\n")

# ==========================================
# ★ 2. 開始批次處理與條件篩選
# ==========================================
for ticker in tickers:
    print(f"正在分析 {ticker} ...")
    
    # 加入 1 秒延遲，保護你的 IP 不被 Yahoo 封鎖
    time.sleep(1) 
    
    df = yf.download(ticker, period="6mo", progress=False)
    
    if df.empty:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    # --- 計算均線與 MACD、RSI、KD (與先前邏輯相同) ---
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['OSC'] = df['MACD'] - df['Signal']

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=13, adjust=False).mean()
    ema_down = down.ewm(com=13, adjust=False).mean()
    rs = ema_up / ema_down
    df['RSI14'] = 100 - (100 / (1 + rs))

    low_9 = df['Low'].rolling(window=9, min_periods=1).min()
    high_9 = df['High'].rolling(window=9, min_periods=1).max()
    df['RSV'] = 100 * (df['Close'] - low_9) / (high_9 - low_9)
    df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()

    df = df.dropna()
    if len(df) < 2:
        continue

    # ==========================================
    # ★ 3. 策略條件篩選
    # ==========================================
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # 判斷邏輯 1：站上月線
    is_above_ma20 = latest['Close'] > latest['MA20']
    # 判斷邏輯 2：KD 黃金交叉 (今日 K>D 且 昨日 K<=D)
    is_kd_golden = (latest['K'] > latest['D']) and (prev['K'] <= prev['D'])

    # 若符合強勢條件，存入專屬名單並保留資料以供畫圖
    if is_above_ma20 and is_kd_golden:
        strong_watchlist.append(f"⭐ {ticker} (收盤價: {latest['Close']:.2f}, 月線: {latest['MA20']:.2f}, K值: {latest['K']:.2f})")
        strong_dfs[ticker] = df # 保留這檔的數據

    # 彙整一般狀態報告
    report = f"【{ticker}】 收盤: {latest['Close']:.2f} | K: {latest['K']:.2f} | D: {latest['D']:.2f} "
    if latest['K'] > 80:
        report += "-> ⚠️ 短線過熱"
    elif latest['K'] < latest['D']:
        report += "-> 📉 KD 偏弱"
    else:
        report += "-> ✅ 多方優勢"
        
    summary_reports.append(report)

# ==========================================
# ★ 4. 顯示文字報告
# ==========================================
print("\n" + "="*50)
print(" 🎯 營建族群強勢觀察名單 (站上月線 + KD金叉) 🎯")
print("="*50)
if len(strong_watchlist) > 0:
    for item in strong_watchlist:
        print(item)
    print("\n💡 稍後將為以上強勢標的自動產生技術線型圖表...")
else:
    print("今日無符合「站上月線且KD黃金交叉」條件的股票。")

print("\n" + "="*50)
print(" 📊 族群狀態總覽 📊")
print("="*50)
for report in summary_reports:
    print(report)

# ==========================================
# ★ 5. 建立文字報告視窗
# ==========================================
fig_text, ax_text = plt.subplots(figsize=(10, max(6, len(summary_reports) * 0.4 + 3)))
ax_text.axis('off')
fig_text.suptitle('營建族群掃描報告', fontsize=14, fontweight='bold')

# 組合報告文字（圖表視窗不支援 Emoji，改用純文字標籤）
lines = []
lines.append('【強勢名單】站上月線 + KD 金叉')
lines.append('=' * 48)
if strong_watchlist:
    for item in strong_watchlist:
        lines.append(item.replace('⭐ ', '[*] '))
else:
    lines.append('今日無符合條件的股票。')
lines.append('')
lines.append('【族群狀態總覽】')
lines.append('=' * 48)
for item in summary_reports:
    clean_item = (item
                  .replace('-> ⚠️ 短線過熱', '-> [過熱]')
                  .replace('-> 📉 KD 偏弱',  '-> [偏弱]')
                  .replace('-> ✅ 多方優勢',  '-> [多方]'))
    lines.append(clean_item)

report_text = '\n'.join(lines)
ax_text.text(0.02, 0.98, report_text, transform=ax_text.transAxes,
             fontsize=9, verticalalignment='top',
             wrap=True)
plt.tight_layout()

# ==========================================
# ★ 6. 專屬繪圖：只畫強勢股的圖表
# ==========================================
if strong_dfs:
    print("\n正在生成強勢股圖表...")
    for tkr, data in strong_dfs.items():
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 13), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
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

# 顯示文字報告視窗與所有強勢股圖表
plt.show()