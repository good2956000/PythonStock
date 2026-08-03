import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# ★ 修正中文顯示
# ==========================================
plt.rcParams['font.sans-serif'] = [
    'Microsoft JhengHei', 'PingFang TC', 'Arial Unicode MS', 'SimHei', 'sans-serif'
]
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# ★ 1. 使用者輸入
# ==========================================
print("=" * 55)
print("   乖離率、布林通道、RSI、MACD、KD 買賣訊號分析")
print("=" * 55)

raw_ticker = input("請輸入股票代號（台股請直接輸入數字，如 2330）：").strip()

# 自動判斷是否為台股代號
if raw_ticker.isdigit():
    ticker = raw_ticker + ".TW"
else:
    ticker = raw_ticker.upper()

period = input("分析期間（預設 1y，可輸入 3mo/6mo/1y/2y/5y）：").strip() or "1y"

print("\n--- 乖離率 (BIAS) 設定 ---")
bias_ma   = int(input("乖離率 MA 天期（預設 20）：").strip() or "20")
bias_sell = float(input("乖離率超過多少 % 視為賣出訊號（預設 +6）：").strip() or "6")
bias_buy  = float(input("乖離率低於多少 % 視為買入訊號（預設 -6）：").strip() or "-6")

print("\n--- 布林通道 (Bollinger Bands) 設定 ---")
bb_ma  = int(input("布林通道 MA 天期（預設 20）：").strip() or "20")
bb_std = float(input("布林通道標準差倍數（預設 2.0）：").strip() or "2.0")

# ==========================================
# ★ 2. 下載資料
# ==========================================
print(f"\n正在下載 {ticker} 資料（{period}）...")
df = yf.download(ticker, period=period, progress=False)

if df.empty:
    print(f"❌ 找不到股票 {ticker}，請確認代號是否正確。")
    exit()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df = df.dropna()

total_days = len(df)
print(f"✅ 共取得 {total_days} 個交易日（{df.index[0].date()} ~ {df.index[-1].date()}）\n")

# ==========================================
# ★ 3. 計算指標
# ==========================================

# --- 乖離率 BIAS ---
df['BIAS_MA'] = df['Close'].rolling(window=bias_ma).mean()
df['BIAS']    = (df['Close'] - df['BIAS_MA']) / df['BIAS_MA'] * 100

# --- 布林通道 Bollinger Bands ---
df['BB_MA']    = df['Close'].rolling(window=bb_ma).mean()
df['BB_STD']   = df['Close'].rolling(window=bb_ma).std()
df['BB_UPPER'] = df['BB_MA'] + bb_std * df['BB_STD']
df['BB_LOWER'] = df['BB_MA'] - bb_std * df['BB_STD']
df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MA'] * 100  # 通道寬度 %

# --- RSI (14 日) ---
delta  = df['Close'].diff()
ema_up = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
ema_dn = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
df['RSI'] = 100 - (100 / (1 + ema_up / ema_dn))

# --- MACD (12/26/9) ---
ema12        = df['Close'].ewm(span=12, adjust=False).mean()
ema26        = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD']   = ema12 - ema26
df['SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['OSC']    = df['MACD'] - df['SIGNAL']

# --- KD (9 日 隨機指標) ---
low9      = df['Low'].rolling(window=9, min_periods=1).min()
high9     = df['High'].rolling(window=9, min_periods=1).max()
hl_range  = (high9 - low9).replace(0, np.nan)
rsv       = 100 * (df['Close'] - low9) / hl_range
df['K']   = rsv.ewm(alpha=1/3, adjust=False).mean()
df['D']   = df['K'].ewm(alpha=1/3, adjust=False).mean()

# --- 均線（出場模組 C/D 需要）---
df['MA5']  = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA60'] = df['Close'].rolling(window=60).mean()

df = df.dropna()

# ==========================================
# ★ 4. 產生買賣訊號
# ==========================================

# BIAS 訊號
df['BIAS_SELL'] = df['BIAS'] >= bias_sell
df['BIAS_BUY']  = df['BIAS'] <= bias_buy

# 布林通道訊號（突破上軌賣、跌破下軌買）
df['BB_SELL'] = df['Close'] >= df['BB_UPPER']
df['BB_BUY']  = df['Close'] <= df['BB_LOWER']

# RSI 訊號
df['RSI_SELL'] = df['RSI'] >= 70
df['RSI_BUY']  = df['RSI'] <= 30

# MACD 黃金/死亡交叉（柱由負轉正 / 正轉負）
osc_prev        = df['OSC'].shift(1)
df['MACD_BUY']  = (df['OSC'] > 0) & (osc_prev <= 0)
df['MACD_SELL'] = (df['OSC'] < 0) & (osc_prev >= 0)

# KD 黃金/死亡交叉（加位置過濾）
k_prev       = df['K'].shift(1)
d_prev       = df['D'].shift(1)
df['KD_BUY']  = (df['K'] > df['D']) & (k_prev <= d_prev) & (df['K'] < 50)
df['KD_SELL'] = (df['K'] < df['D']) & (k_prev >= d_prev) & (df['K'] > 50)

# 右側順勢：站上季線 + 月線多頭 + MACD 金叉 + 正乖離 <= 5%（防追高）
is_uptrend     = df['Close'] > df['MA60']
bias_from_ma20 = (df['Close'] - df['BIAS_MA']) / df['BIAS_MA'] * 100
df['COMBO_BUY'] = is_uptrend & (df['Close'] > df['BIAS_MA']) & df['MACD_BUY'] & (bias_from_ma20 <= 5)

# ==========================================
# ★ 4.5 動態出場策略模擬
# ==========================================
class _PositionState:
    def __init__(self, entry_price, entry_low):
        self.entry_price    = entry_price
        self.entry_low      = entry_low
        self.highest_price  = entry_price
        self.is_overbought  = False
        self.macd_activated = False  # 等首次 OSC 翻正後才啟動模組 B

class _ExitStrategy:
    def __init__(self, stop_loss_pct=0.07, profit_protect_pct=0.05):
        self.stop_loss_pct      = stop_loss_pct
        self.profit_protect_pct = profit_protect_pct
        self.position           = None

    def on_enter(self, price, low):
        self.position = _PositionState(price, low)

    def check_exit(self, cur, prev, p2):
        if not self.position:
            return False, ""
        c = cur['Close']
        if c > self.position.highest_price:
            self.position.highest_price = c
        if cur['K'] >= 80:
            self.position.is_overbought = True

        # A 初始停損
        if c < self.position.entry_low:
            return True, "A 初始停損：跌破進場K線低點"
        if (c - self.position.entry_price) / self.position.entry_price <= -self.stop_loss_pct:
            return True, f"A 初始停損：帳面虧損達 {self.stop_loss_pct*100:.0f}%"

        # B 動能衰退（只看實質死叉，拿掉易誤判的紅柱縮短）
        if not self.position.macd_activated and cur['OSC'] > 0:
            self.position.macd_activated = True
        if self.position.macd_activated:
            if cur['OSC'] < 0 and prev['OSC'] >= 0:
                return True, "B 動能衰退：MACD 死亡交叉"

        # C 高檔鈍化破位（只看 KD 高檔破位，拿掉太敏感的 MA5）
        if self.position.is_overbought:
            if cur['K'] < 80 and prev['K'] >= 80:
                return True, "C 鈍化破位：K值高檔跌破 80"

        # D 移動停利（防守線退後到 MA20 月線，確保吃到大波段）
        if (c - self.position.entry_price) / self.position.entry_price > self.profit_protect_pct:
            if c < cur.get('MA20', cur['MA10']):
                return True, "D 移動停利：獲利保護，跌破 20 日月線"

        return False, ""

# 歷史回測模擬（逐日掃描買進→出場）
df['EXIT_SIGNAL']  = False
df['EXIT_REASON']  = ''
exit_strategy     = _ExitStrategy()
in_position       = False
trades            = []    # (entry_price, exit_price)
entry_price_track = None

for i in range(2, len(df)):
    cur  = df.iloc[i].to_dict()
    prev = df.iloc[i - 1].to_dict()
    p2   = df.iloc[i - 2].to_dict()

    if not in_position and bool(cur['COMBO_BUY']):
        exit_strategy.on_enter(cur['Close'], cur['Low'])
        in_position       = True
        entry_price_track = cur['Close']
    elif in_position:
        is_exit, reason = exit_strategy.check_exit(cur, prev, p2)
        if is_exit:
            df.at[df.index[i], 'EXIT_SIGNAL'] = True
            df.at[df.index[i], 'EXIT_REASON'] = reason
            trades.append((entry_price_track, cur['Close']))
            in_position       = False
            exit_strategy.position = None
            entry_price_track = None

# ==========================================
# ★ 5. 印出最近訊號摘要
# ==========================================
latest = df.iloc[-1]
print("=" * 60)
print(f"【{ticker}】最新狀態（{df.index[-1].date()}）  策略：右側順勢（趨勢流）")
print("=" * 60)
print(f"  收盤價              ：{float(latest['Close']):.2f}")
print(f"  BIAS({bias_ma}日)        ：{float(latest['BIAS']):+.2f}%  （賣≥+{bias_sell}% / 買≤{bias_buy}%）")
print(f"  布林 上/中/下軌      ：{float(latest['BB_UPPER']):.2f} / {float(latest['BB_MA']):.2f} / {float(latest['BB_LOWER']):.2f}")
print(f"  RSI(14)             ：{float(latest['RSI']):.1f}  （超買≥70 / 超賣≤30）")
print(f"  MACD / Signal / OSC ：{float(latest['MACD']):.4f} / {float(latest['SIGNAL']):.4f} / {float(latest['OSC']):+.4f}")
print(f"  K / D               ：{float(latest['K']):.1f} / {float(latest['D']):.1f}")
print()

# 判斷目前訊號
close_val = float(latest['Close'])
bias_val  = float(latest['BIAS'])
rsi_val   = float(latest['RSI'])
k_val     = float(latest['K'])
osc_val   = float(latest['OSC'])
osc_prev2 = float(df['OSC'].iloc[-2])

signals = []
if bias_val >= bias_sell:
    signals.append(("賣", f"BIAS 乖離率 {bias_val:+.2f}% ≥ +{bias_sell}%"))
if bias_val <= bias_buy:
    signals.append(("買", f"BIAS 乖離率 {bias_val:+.2f}% ≤ {bias_buy}%"))
if close_val >= float(latest['BB_UPPER']):
    signals.append(("賣", f"收盤 {close_val:.2f} 突破布林上軌 {float(latest['BB_UPPER']):.2f}"))
if close_val <= float(latest['BB_LOWER']):
    signals.append(("買", f"收盤 {close_val:.2f} 跌破布林下軌 {float(latest['BB_LOWER']):.2f}"))
if rsi_val >= 70:
    signals.append(("賣", f"RSI {rsi_val:.1f} ≥ 70（超買）"))
if rsi_val <= 30:
    signals.append(("買", f"RSI {rsi_val:.1f} ≤ 30（超賣）"))
if osc_val > 0 and osc_prev2 <= 0:
    signals.append(("買", "MACD 柱狀由負轉正（黃金交叉）"))
if osc_val < 0 and osc_prev2 >= 0:
    signals.append(("賣", "MACD 柱狀由正轉負（死亡交叉）"))
if k_val <= 20:
    signals.append(("買", f"KD K值 {k_val:.1f} 進入超賣區（≤20）"))
if k_val >= 80:
    signals.append(("賣", f"KD K值 {k_val:.1f} 進入超買區（≥80）"))

buy_sigs  = [msg for typ, msg in signals if typ == "買"]
sell_sigs = [msg for typ, msg in signals if typ == "賣"]

if signals:
    print("【當前訊號】")
    for msg in sell_sigs:
        print(f"  ⚠️  {msg}")
    for msg in buy_sigs:
        print(f"  ✅  {msg}")
    if len(buy_sigs) >= 3:
        print(f"\n  🔥 強力買入訊號：{len(buy_sigs)} 個指標同時觸發！")
    if len(sell_sigs) >= 3:
        print(f"\n  🚨 強力賣出訊號：{len(sell_sigs)} 個指標同時觸發！")
else:
    print("  📊 目前無明確買賣訊號，股價位於中性區。")

# 近期訊號紀錄
print()
sig_mask = (df['BIAS_SELL'] | df['BIAS_BUY'] | df['BB_SELL']   | df['BB_BUY']  |
            df['RSI_SELL']  | df['RSI_BUY']  | df['MACD_SELL'] | df['MACD_BUY'] |
            df['KD_SELL']   | df['KD_BUY']   | df['COMBO_BUY'] | df['EXIT_SIGNAL'])
recent_signals = df[sig_mask].tail(15)
if not recent_signals.empty:
    print("【近期關鍵訊號紀錄 (含動態出場)】")
    print(f"  {'日期':<12} {'收盤':>7} {'RSI':>5} {'K':>5} {'D':>5}  訊號/出場原因")
    print("  " + "-" * 75)
    for idx, row in recent_signals.iterrows():
        tags = []
        if row['COMBO_BUY']:
            tags.append("🔥強力買進(MACD確認)")
        elif row['EXIT_SIGNAL']:
            tags.append(f"🛑出場({row['EXIT_REASON']})")
        else:
            if row['BIAS_BUY']:   tags.append("BIAS買")
            if row['RSI_BUY']:    tags.append("RSI買")
            if row['MACD_BUY']:   tags.append("MACD金叉")
            if row['KD_BUY']:     tags.append("KD金叉")
            if row['MACD_SELL']:  tags.append("MACD死叉")
        print(f"  {str(idx.date()):<12} {float(row['Close']):>7.2f} "
              f"{float(row['RSI']):>5.1f} "
              f"{float(row['K']):>5.1f} "
              f"{float(row['D']):>5.1f}  {'、'.join(tags)}")

print()

# 回測績效統計
if trades:
    ret_list = [(ep - ent) / ent * 100 for ent, ep in trades]
    wins     = [r for r in ret_list if r > 0]
    losses   = [r for r in ret_list if r <= 0]
    win_rate = len(wins) / len(trades) * 100
    avg_win  = sum(wins)   / len(wins)   if wins   else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    rr_ratio = abs(avg_win / avg_loss)   if avg_loss != 0 else float('inf')
    print("【回測績效統計】")
    print(f"  策略路線            ：右側順勢（趨勢流）")
    print(f"  總交易次數          ：{len(trades)} 筆")
    print(f"  勝率                ：{win_rate:.1f}%  （{len(wins)} 勝 / {len(losses)} 敗）")
    print(f"  平均獲利            ：{avg_win:+.2f}%")
    print(f"  平均虧損            ：{avg_loss:+.2f}%")
    print(f"  賺賠比              ：{rr_ratio:.2f}  （>1 代表期望值為正）")
else:
    print("【回測績效統計】無完整進出場紀錄（期間內無進出場或尚未出場）")
print()

# ==========================================
# ★ 6. 繪圖（6 格圖表）
# ==========================================
LOC = 'upper left'  # legend 位置常數
fig, axes = plt.subplots(6, 1, figsize=(14, 20),
                         gridspec_kw={'height_ratios': [3, 1.2, 1.2, 1.5, 1.2, 0.8]})
fig.suptitle(f"{ticker}  技術指標綜合分析\n"
             f"BIAS / 布林通道 / RSI / MACD / KD  "
             f"（{df.index[0].date()} ~ {df.index[-1].date()}）",
             fontsize=13, fontweight='bold')

dates = df.index

# ── 圖1：收盤價 + 布林通道 + 綜合強訊號 ──
ax1 = axes[0]
ax1.plot(dates, df['Close'],    color='#333333', linewidth=1.2, label='收盤價', zorder=3)
ax1.plot(dates, df['BB_MA'],    color='#2196F3', linewidth=1,   linestyle='--', label=f'中軌 MA{bb_ma}', alpha=0.8)
ax1.plot(dates, df['BB_UPPER'], color='#F44336', linewidth=1,   label=f'上軌 +{bb_std}σ', alpha=0.7)
ax1.plot(dates, df['BB_LOWER'], color='#4CAF50', linewidth=1,   label=f'下軌 -{bb_std}σ', alpha=0.7)
ax1.fill_between(dates, df['BB_UPPER'], df['BB_LOWER'], alpha=0.06, color='#2196F3')
combo_buy_d  = df[df['COMBO_BUY']].index
exit_sig_d   = df[df['EXIT_SIGNAL']].index
ax1.scatter(combo_buy_d,  df.loc[combo_buy_d,  'Close'], marker='^',
            color='#1B5E20', s=130, zorder=6, label='進場訊號(▲)')
ax1.scatter(exit_sig_d, df.loc[exit_sig_d, 'Close'], marker='v',
            color='#B71C1C', s=130, zorder=7, label='動態出場觸發(▼)')
ax1.set_ylabel('價格', fontsize=10)
ax1.legend(loc=LOC, fontsize=8, ncol=4)
ax1.grid(True, alpha=0.3)
ax1.set_title('布林通道 + 訊號（▲右側順勢：月線多頭+MACD金叉+季線保護  ▼動態停利停損出場）', fontsize=11)

# ── 圖2：乖離率 BIAS ──
ax2 = axes[1]
bias_colors = ['#F44336' if v >= bias_sell else '#4CAF50' if v <= bias_buy else '#90A4AE'
               for v in df['BIAS']]
ax2.bar(dates, df['BIAS'], color=bias_colors, width=1, alpha=0.8)
ax2.axhline(y=bias_sell, color='#F44336', linewidth=1.5, linestyle='--', label=f'賣 +{bias_sell}%')
ax2.axhline(y=bias_buy,  color='#4CAF50', linewidth=1.5, linestyle='--', label=f'買 {bias_buy}%')
ax2.axhline(y=0, color='#555', linewidth=0.8)
ax2.legend(loc=LOC, fontsize=8)
ax2.set_ylabel(f'BIAS({bias_ma})%', fontsize=9)
ax2.set_title(f'乖離率 BIAS ({bias_ma}日)', fontsize=10)
ax2.grid(True, alpha=0.3)

# ── 圖3：RSI ──
ax3 = axes[2]
ax3.plot(dates, df['RSI'], color='#9C27B0', linewidth=1.2, label='RSI(14)')
ax3.axhline(y=70, color='#F44336', linewidth=1.2, linestyle='--', label='超買 70')
ax3.axhline(y=30, color='#4CAF50', linewidth=1.2, linestyle='--', label='超賣 30')
ax3.axhline(y=50, color='#aaa', linewidth=0.8, linestyle=':')
ax3.fill_between(dates, 70, df['RSI'].clip(upper=100),
                 where=(df['RSI'] >= 70), alpha=0.15, color='#F44336')
ax3.fill_between(dates, df['RSI'].clip(lower=0), 30,
                 where=(df['RSI'] <= 30), alpha=0.15, color='#4CAF50')
ax3.set_ylim(0, 100)
ax3.set_ylabel('RSI', fontsize=9)
ax3.legend(loc=LOC, fontsize=8, ncol=3)
ax3.set_title('RSI (14日)  超買>=70 / 超賣<=30', fontsize=10)
ax3.grid(True, alpha=0.3)

# ── 圖4：MACD ──
ax4 = axes[3]
osc_colors = ['#F44336' if v >= 0 else '#4CAF50' for v in df['OSC']]
ax4.bar(dates, df['OSC'],    color=osc_colors, width=1, alpha=0.7, label='OSC 柱狀')
ax4.plot(dates, df['MACD'],  color='#2196F3', linewidth=1.2, label='MACD')
ax4.plot(dates, df['SIGNAL'],color='#FF9800', linewidth=1.2, label='Signal')
ax4.axhline(y=0, color='#555', linewidth=0.8)
macd_buy_d  = df[df['MACD_BUY']].index
macd_sell_d = df[df['MACD_SELL']].index
ax4.scatter(macd_buy_d,  df.loc[macd_buy_d,  'MACD'], marker='^',
            color='#4CAF50', s=70, zorder=5, label='金叉')
ax4.scatter(macd_sell_d, df.loc[macd_sell_d, 'MACD'], marker='v',
            color='#F44336', s=70, zorder=5, label='死叉')
ax4.legend(loc=LOC, fontsize=8, ncol=5)
ax4.set_ylabel('MACD', fontsize=9)
ax4.set_title('MACD (12/26/9)  ▲金叉 ▼死叉', fontsize=10)
ax4.grid(True, alpha=0.3)

# ── 圖5：KD ──
ax5 = axes[4]
ax5.plot(dates, df['K'], color='#2196F3', linewidth=1.2, label='K')
ax5.plot(dates, df['D'], color='#FF9800', linewidth=1.2, label='D')
ax5.axhline(y=80, color='#F44336', linewidth=1.2, linestyle='--', label='超買 80')
ax5.axhline(y=20, color='#4CAF50', linewidth=1.2, linestyle='--', label='超賣 20')
ax5.fill_between(dates, 80, 100, alpha=0.06, color='#F44336')
ax5.fill_between(dates, 0,  20,  alpha=0.06, color='#4CAF50')
kd_buy_d  = df[df['KD_BUY']].index
kd_sell_d = df[df['KD_SELL']].index
ax5.scatter(kd_buy_d,  df.loc[kd_buy_d,  'K'], marker='^', color='#4CAF50', s=60, zorder=5, label='KD金叉')
ax5.scatter(kd_sell_d, df.loc[kd_sell_d, 'K'], marker='v', color='#F44336', s=60, zorder=5, label='KD死叉')
ax5.set_ylim(0, 100)
ax5.legend(loc=LOC, fontsize=8, ncol=3)
ax5.set_ylabel('KD', fontsize=9)
ax5.set_title('KD 隨機指標 (9日)  ▲金叉(K<50) ▼死叉(K>50)', fontsize=10)
ax5.grid(True, alpha=0.3)

# ── 圖6：成交量 ──
ax6 = axes[5]
vol_colors = ['#F44336' if c >= o else '#4CAF50'
              for c, o in zip(df['Close'], df['Open'])]
ax6.bar(dates, df['Volume'], color=vol_colors, width=1, alpha=0.7)
ax6.set_ylabel('成交量', fontsize=9)
ax6.set_title('成交量', fontsize=10)
ax6.grid(True, alpha=0.3)

# 格式化 x 軸
for ax in axes:
    ax.set_xlim(dates[0], dates[-1])
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig(f"{raw_ticker}_technical_analysis.png", dpi=150, bbox_inches='tight')
print(f"📊 圖表已儲存：{raw_ticker}_technical_analysis.png")
plt.show()
