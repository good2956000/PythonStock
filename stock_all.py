import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 獲取歷史市場數據
# ==========================================
start_date = "2020-06-01"
# 抓取大盤、0050 的日K線資料 (Open/High/Low/Close)，
# 大盤額外需要高低價算KD，0050 則額外保留 Open 以繪製日K線圖
twii = yf.download("^TWII", start=start_date)[['High', 'Low', 'Close']]
twii.columns = ['High', 'Low', 'Close']

ohlc_0050 = yf.download("0050.TW", start=start_date)[['Open', 'High', 'Low', 'Close']]
ohlc_0050.columns = ['Open', 'High', 'Low', 'Close']

# 合併數據並剔除空值 (確保日期對齊)
df = pd.DataFrame({
    'TWII_H': twii['High'],
    'TWII_L': twii['Low'],
    'TWII_C': twii['Close'],
    'ETF_0050': ohlc_0050['Close'],
}).dropna()

# ==========================================
# 2. 計算大盤 KD 指標 (參數: 9, 3, 3)
# ==========================================
# 計算 9 日 RSV
low_min = df['TWII_L'].rolling(window=9).min()
high_max = df['TWII_H'].rolling(window=9).max()
df['RSV'] = 100 * (df['TWII_C'] - low_min) / (high_max - low_min)

# 計算 K 值與 D 值 (com=2 等同於平滑係數 1/3)
df['K'] = df['RSV'].ewm(com=2, adjust=False).mean()
df['D'] = df['K'].ewm(com=2, adjust=False).mean()

# ==========================================
# 3. 建立交易訊號與部位轉換 (優化版)
# ==========================================
# 1. 準備基礎數據：計算 20 日均線與乖離率 (Bias)
df['SMA_20'] = df['TWII_C'].rolling(window=20).mean()
df['Bias_20'] = (df['TWII_C'] - df['SMA_20']) / df['SMA_20'] * 100

# 2. 買進條件：維持原本的低檔買入，勝率最高
df['Buy_Signal'] = (df['K'].shift(1) < 20) & (df['K'].shift(1) <= df['D'].shift(1)) & (df['K'] > df['D'])

# 3. 定義兩種賣出情境
kd_death_cross = (df['K'].shift(1) > 80) & (df['K'].shift(1) >= df['D'].shift(1)) & (df['K'] < df['D'])

# 賣出條件一 (極度樂觀時的停利)：KD高檔死叉 且 大盤正乖離大於 3.5%
# (3.5% 對加權指數來說已是極大的短線過熱)
sell_take_profit = kd_death_cross & (df['Bias_20'] > 3.5)

# 賣出條件二 (趨勢轉弱時的防守)：收盤「剛跌破」月線 且 K值跌破 50
# (加入昨天≥月線、今天<月線的邊緣觸發判斷，避免跌破後每天重複觸發；
#  再疊加 K<50 雙重確認，避免盤整時均線糾結被洗出場)
sell_trend_break = (
    (df['TWII_C'] < df['SMA_20']) &
    (df['TWII_C'].shift(1) >= df['SMA_20'].shift(1)) &
    (df['K'] < 50)
)

# 4. 綜合賣出訊號：只要滿足其中一項就賣出
df['Sell_Signal'] = sell_take_profit | sell_trend_break

# 標記訊號狀態：買進為 1，賣出為 -1，其餘為 0
df['Signal'] = 0
df.loc[df['Buy_Signal'], 'Signal'] = 1
df.loc[df['Sell_Signal'], 'Signal'] = -1

# 填補持倉狀態 (遇到 1 就持有，遇到 -1 就空手)
df['Position'] = df['Signal'].replace(0, np.nan).ffill().replace(-1, 0).fillna(0)

# 遞延一天：訊號出現的隔天才會反映在報酬上
df['Position'] = df['Position'].shift(1).fillna(0)

# ==========================================
# 4. 計算策略報酬率與勝率相關數據
# ==========================================
# 計算每日漲跌幅
df['0050_Daily_Ret'] = df['ETF_0050'].pct_change()

# 計算策略實際吃到的報酬 (每日報酬 * 持倉狀態)
df['Strategy_0050_Ret'] = df['0050_Daily_Ret'] * df['Position']

# 計算累積報酬率
df['0050_Cum_Ret'] = (1 + df['Strategy_0050_Ret']).cumprod()

# 簡單印出最終累積報酬結果
print(f"--- KD波段策略回測結果 ({start_date} 至今) ---")
print(f"0050 累積報酬率: {(df['0050_Cum_Ret'].iloc[-1] - 1) * 100:.2f}%")

# 計算交易次數與簡單勝率 (以每次完整買賣區間計算)
trades = df[df['Signal'] != 0].copy()
print(f"總觸發進/出場訊號次數: {len(trades)}")

# ==========================================
# 5. 列出每筆進出場時間點
# ==========================================
# 注意：Position 於第 61 行遞延一天生效，因此「訊號日」的隔一個交易日
# 才是實際進場/出場、開始吃到報酬的日期
print(f"\n--- 每筆進出場明細 ({len(trades)} 筆訊號) ---")
print(f"{'方向':<6}{'訊號日':<12}{'實際執行日':<12}{'0050價':>10}")

records = []
for signal_date, row in trades.iterrows():
    loc = df.index.get_loc(signal_date)
    if loc + 1 >= len(df):
        print(f"{'進場' if row['Signal'] == 1 else '出場':<6}{signal_date.strftime('%Y-%m-%d'):<12}{'(尚無隔日資料)':<12}")
        continue
    exec_date = df.index[loc + 1]
    direction = '進場' if row['Signal'] == 1 else '出場'
    exec_0050 = df.loc[exec_date, 'ETF_0050']
    print(f"{direction:<6}{signal_date.strftime('%Y-%m-%d'):<12}{exec_date.strftime('%Y-%m-%d'):<12}{exec_0050:>10.2f}")
    records.append({
        'exec_date': exec_date,
        'direction': row['Signal'],
        'exec_0050': exec_0050,
    })

# 配對進場/出場，計算每輪交易的持有天數與報酬率
print("\n--- 每輪交易 (進場 → 出場) 報酬 ---")
entry = None
for rec in records:
    if rec['direction'] == 1 and entry is None:
        entry = rec
    elif rec['direction'] == -1 and entry is not None:
        hold_days = (rec['exec_date'] - entry['exec_date']).days
        ret_0050 = (rec['exec_0050'] / entry['exec_0050'] - 1) * 100
        print(
            f"進場 {entry['exec_date'].strftime('%Y-%m-%d')} → 出場 {rec['exec_date'].strftime('%Y-%m-%d')} "
            f"(持有 {hold_days} 天)  0050報酬: {ret_0050:+.2f}%"
        )
        entry = None

if entry is not None:
    print(f"進場 {entry['exec_date'].strftime('%Y-%m-%d')} → 目前尚持有中 (未出場)")

# ==========================================
# 6. 互動式日K線圖 (Plotly)：可縮放拖曳，標示進出場點
# ==========================================
buy_dates = [r['exec_date'] for r in records if r['direction'] == 1]
sell_dates = [r['exec_date'] for r in records if r['direction'] == -1]

plot_0050 = ohlc_0050.loc[df.index]

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=plot_0050.index, open=plot_0050['Open'], high=plot_0050['High'],
    low=plot_0050['Low'], close=plot_0050['Close'],
    increasing_line_color='#F44336', decreasing_line_color='#4CAF50',
    name='0050', showlegend=False,
))

if buy_dates:
    fig.add_trace(go.Scatter(
        x=buy_dates, y=plot_0050.loc[buy_dates, 'Low'] * 0.97,
        mode='markers', marker={'symbol': 'triangle-up', 'size': 12, 'color': '#1565C0'},
        name='進場(買進)',
    ))

if sell_dates:
    fig.add_trace(go.Scatter(
        x=sell_dates, y=plot_0050.loc[sell_dates, 'High'] * 1.03,
        mode='markers', marker={'symbol': 'triangle-down', 'size': 12, 'color': '#6A1B9A'},
        name='出場(賣出)',
    ))

# 拿掉內建 rangeslider，保留滑鼠拖曳平移 + 滾輪/框選縮放的互動能力
fig.update_xaxes(rangeslider_visible=False)
fig.update_layout(
    title=f"0050 大盤KD波段策略 進出場點位 ({start_date} 至今)",
    height=700, hovermode='x unified',
    yaxis_title='價格',
    font={'family': 'Microsoft JhengHei, PingFang TC, Arial Unicode MS, sans-serif'},
)

fig.write_html("kd_strategy_signals.html")
print("\n[圖表] 互動式K線進出場點位圖已儲存：kd_strategy_signals.html (可用瀏覽器開啟縮放拖曳)")
fig.show()
