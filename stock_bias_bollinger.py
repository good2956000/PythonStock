import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
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
print("   RSI、MACD、KD 買賣訊號分析")
print("=" * 55)

raw_input_str = input("請輸入股票代號（可輸入單一或多檔，以逗號/空白分隔，如 2330 2454 6180）：").strip()
# 支援半形/全形逗號、頓號、空白作為多檔代號的分隔符
raw_tickers = [t for t in re.split(r'[,，、\s]+', raw_input_str) if t]

period = input("分析期間（預設 1y，可輸入 3mo/6mo/1y/2y/5y）：").strip() or "1y"


def analyze_stock(raw_ticker, period):
    """對單一股票代號執行完整的技術指標分析、回測與繪圖。"""
    # 自動判斷是否為台股代號（上市 .TW / 上櫃 .TWO 由下載階段自動偵測）
    if raw_ticker.isdigit():
        ticker = raw_ticker + ".TW"
    else:
        ticker = raw_ticker.upper()

    # ==========================================
    # ★ 2. 下載資料
    # ==========================================
    print(f"\n正在下載 {ticker} 資料（{period}）...")
    df = yf.download(ticker, period=period, progress=False)

    # 上市（.TW）查無資料時，自動改試上櫃（.TWO）
    if df.empty and raw_ticker.isdigit():
        ticker = raw_ticker + ".TWO"
        print(f"⚠️  上市代號查無資料，改嘗試上櫃 {ticker} ...")
        df = yf.download(ticker, period=period, progress=False)

    if df.empty:
        print(f"❌ 找不到股票 {ticker}，請確認代號是否正確。")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df = df.dropna()

    total_days = len(df)
    print(f"✅ 共取得 {total_days} 個交易日（{df.index[0].date()} ~ {df.index[-1].date()}）\n")

    # ==========================================
    # ★ 3. 計算指標
    # ==========================================

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

    # --- 成交量均量（量能過濾用）---
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()

    df = df.dropna()

    # ==========================================
    # ★ 4. 產生買賣訊號
    # ==========================================

    # RSI 訊號
    df['RSI_SELL'] = (df['RSI'].shift(1) > 70) & (df['RSI'] <= 70)  # RSI 跌破 70（前一天>70）
    df['RSI_BUY']  = df['RSI'] <= 30

    # MACD 黃金/死亡交叉（柱由負轉正 / 正轉負）
    osc_prev        = df['OSC'].shift(1)
    df['MACD_BUY']  = (df['OSC'] > 0) & (osc_prev <= 0)
    df['MACD_SELL'] = (df['OSC'] < 0) & (osc_prev >= 0)

    # OSC 小於 0 (還在水下)，但今天的值大於昨天 (空方動能開始收斂)
    df['OSC_SHRINK'] = (df['OSC'] < 0) & (df['OSC'] > df['OSC'].shift(1))

    # KD 黃金/死亡交叉（加位置過濾）
    k_prev       = df['K'].shift(1)
    d_prev       = df['D'].shift(1)
    df['KD_BUY']  = (df['K'] > df['D']) & (k_prev <= d_prev) & (df['K'] < 50)
    df['KD_SELL'] = (df['K'] < df['D']) & (k_prev >= d_prev) & (df['K'] > 50)

    # 複合賣出：RSI 跌破 70 + KD 高檔死叉（同日觸發）
    df['RSI_KD_SELL'] = df['RSI_SELL'] & df['KD_SELL']

    # OSC 動態 0 軸判定：依近 60 日 OSC 波動幅度的 5% 定義「貼近 0 軸」區間，抓即將翻多的早期訊號
    osc_max      = df['OSC'].rolling(60).max()
    osc_min      = df['OSC'].rolling(60).min()
    osc_range    = osc_max - osc_min
    zero_zone    = osc_range * 0.05
    near_zero    = df['OSC'].abs() <= zero_zone
    momentum_up  = df['OSC'] > df['OSC'].shift(1)
    df['OSC_READY'] = near_zero & momentum_up

    # 右側順勢：計分制取代一票否決（MA60/月線突破瞬間為價格核心，各佔 40 分；OSC/量能為輔助，各佔 10 分）
    long_trend_ok   = df['Close'] > df['MA60']
    # 嚴格定義黃金交叉瞬間：昨天≤月線、今天>月線，避免多頭行情中每天重複觸發
    cross_over_ma20 = (df['Close'] > df['MA20']) & (df['Close'].shift(1) <= df['MA20'].shift(1))
    df['Bias_20']   = (df['Close'] - df['MA20']) / df['MA20']
    volume_ok       = df['Volume'] > df['Volume_MA5']
    df['ENTRY_SCORE'] = (
        long_trend_ok.astype(int)   * 40 +
        cross_over_ma20.astype(int) * 40 +
        df['OSC_READY'].astype(int) * 10 +
        volume_ok.astype(int)       * 10
    )
    df['COMBO_BUY'] = df['ENTRY_SCORE'] >= 80

    # 回檔買入：拿掉 MA20 限制，改看季線乖離（抓回踩季線支撐的低接機會）
    df['Bias_60'] = (df['Close'] - df['MA60']) / df['MA60']
    df['PULLBACK_BUY'] = (
        long_trend_ok &                     # 依然要站上季線（長多格局不變）
        df['OSC_READY'] &                   # 動態 OSC 貼近 0 軸且動能向上
        (df['Bias_60'] > 0) &                # 確保在季線之上（相對於季線是正乖離）
        (df['Bias_60'] <= 0.05) &            # 距離季線不超過 5%（確保買在季線附近的支撐區）
        volume_ok
    )

    # 進場總訊號：站上月線強勢突破 或 回踩季線支撐
    df['ENTRY_SIGNAL'] = df['COMBO_BUY'] | df['PULLBACK_BUY']

    # ==========================================
    # ★ 4.5 動態出場策略模擬
    # ==========================================
    class _PositionState:
        def __init__(self, entry_price, entry_low):
            self.entry_price   = entry_price
            self.entry_low     = entry_low
            self.highest_price = entry_price

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

            # A 初始停損
            if c < self.position.entry_low:
                return True, "A 初始停損：跌破進場K線低點"
            if (c - self.position.entry_price) / self.position.entry_price <= -self.stop_loss_pct:
                return True, f"A 初始停損：帳面虧損達 {self.stop_loss_pct*100:.0f}%"

            # 獲利超過 5% 後啟動保護：跌破 MA10 或創近 3 日收盤新低即停利
            if (c - self.position.entry_price) / self.position.entry_price > self.profit_protect_pct:
                three_day_low = c < min(prev['Close'], p2['Close'])
                if c < cur['MA10'] or three_day_low:
                    return True, "停利保護：跌破MA10或創近3日新低"

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

        if not in_position and bool(cur['ENTRY_SIGNAL']):
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
    print(f"  RSI(14)             ：{float(latest['RSI']):.1f}  （超買≥70 / 超賣≤30）")
    print(f"  MACD / Signal / OSC ：{float(latest['MACD']):.4f} / {float(latest['SIGNAL']):.4f} / {float(latest['OSC']):+.4f}")
    print(f"  K / D               ：{float(latest['K']):.1f} / {float(latest['D']):.1f}")
    print()

    # 判斷目前訊號
    rsi_val      = float(latest['RSI'])
    rsi_prev_val = float(df['RSI'].iloc[-2])
    k_val        = float(latest['K'])
    osc_val      = float(latest['OSC'])
    osc_prev2    = float(df['OSC'].iloc[-2])

    signals = []
    if rsi_prev_val > 70 and rsi_val <= 70 and bool(df['KD_SELL'].iloc[-1]):
        signals.append(("賣", f"RSI 跌破70（{rsi_prev_val:.1f}→{rsi_val:.1f}）且 KD 高檔死叉"))
    if rsi_val <= 30:
        signals.append(("買", f"RSI {rsi_val:.1f} ≤ 30（超賣）"))
    if osc_val > 0 and osc_prev2 <= 0:
        signals.append(("買", "MACD 柱狀由負轉正（黃金交叉）"))
    if osc_val < 0 and osc_prev2 >= 0:
        signals.append(("賣", "MACD 柱狀由正轉負（死亡交叉）"))
    if osc_val < 0 and osc_val > osc_prev2:
        signals.append(("觀察", f"MACD OSC 水下收斂 ({osc_prev2:+.4f} → {osc_val:+.4f})，空方動能衰退"))
    if k_val <= 20:
        signals.append(("買", f"KD K值 {k_val:.1f} 進入超賣區（≤20）"))

    buy_sigs   = [msg for typ, msg in signals if typ == "買"]
    sell_sigs  = [msg for typ, msg in signals if typ == "賣"]
    watch_sigs = [msg for typ, msg in signals if typ == "觀察"]

    if signals:
        print("【當前訊號】")
        for msg in sell_sigs:
            print(f"  ⚠️  {msg}")
        for msg in buy_sigs:
            print(f"  ✅  {msg}")
        for msg in watch_sigs:
            print(f"  🔍  {msg}")
        if len(buy_sigs) >= 3:
            print(f"\n  🔥 強力買入訊號：{len(buy_sigs)} 個指標同時觸發！")
        if len(sell_sigs) >= 3:
            print(f"\n  🚨 強力賣出訊號：{len(sell_sigs)} 個指標同時觸發！")
    else:
        print("  📊 目前無明確買賣訊號，股價位於中性區。")

    # 近期訊號紀錄
    print()
    sig_mask = (df['RSI_SELL']    | df['RSI_BUY']    | df['MACD_SELL']    | df['MACD_BUY']  |
                df['KD_SELL']     | df['KD_BUY']     | df['ENTRY_SIGNAL'] | df['EXIT_SIGNAL'] |
                df['OSC_SHRINK']  | df['RSI_KD_SELL'])
    recent_signals = df[sig_mask].tail(15)
    if not recent_signals.empty:
        print("【近期關鍵訊號紀錄 (含動態出場)】")
        print(f"  {'日期':<12} {'收盤':>7} {'RSI':>5} {'K':>5} {'D':>5}  訊號/出場原因")
        print("  " + "-" * 75)
        for idx, row in recent_signals.iterrows():
            tags = []
            if row['COMBO_BUY']:
                tags.append("🔥強勢突破買進")
            elif row['PULLBACK_BUY']:
                tags.append("📈回踩季線買進")
            elif row['EXIT_SIGNAL']:
                tags.append(f"🛑出場({row['EXIT_REASON']})")
            else:
                if row['RSI_BUY']:     tags.append("RSI買")
                if row['MACD_BUY']:    tags.append("MACD金叉")
                if row['KD_BUY']:      tags.append("KD金叉")
                if row['MACD_SELL']:    tags.append("MACD死叉")
                if row['RSI_KD_SELL']:  tags.append("🚨RSI破70+KD死叉")
                if row['OSC_SHRINK']:   tags.append("🔍OSC水下收斂")
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
    # ★ 6. 繪圖（5 格圖表）
    # ==========================================
    LOC = 'upper left'  # legend 位置常數
    fig, axes = plt.subplots(5, 1, figsize=(14, 17),
                             gridspec_kw={'height_ratios': [3, 1.2, 1.5, 1.2, 0.8]})
    fig.suptitle(f"{ticker}  技術指標綜合分析\n"
                 f"RSI / MACD / KD  "
                 f"（{df.index[0].date()} ~ {df.index[-1].date()}）",
                 fontsize=13, fontweight='bold')

    dates = df.index

    # ── 圖1：收盤價 + MA20/MA60 + 訊號 ──
    ax1 = axes[0]
    ax1.plot(dates, df['Close'], color='#333333', linewidth=1.2, label='收盤價', zorder=3)
    ax1.plot(dates, df['MA20'],  color='#2196F3', linewidth=1,   linestyle='--', label='MA20', alpha=0.8)
    ax1.plot(dates, df['MA60'],  color='#FF9800', linewidth=1,   linestyle='--', label='MA60', alpha=0.8)
    combo_buy_d    = df[df['COMBO_BUY']].index
    pullback_buy_d = df[df['PULLBACK_BUY'] & ~df['COMBO_BUY']].index
    exit_sig_d     = df[df['EXIT_SIGNAL']].index
    ax1.scatter(combo_buy_d,  df.loc[combo_buy_d,  'Close'], marker='^',
                color='#1B5E20', s=130, zorder=6, label='強勢突破(▲)')
    ax1.scatter(pullback_buy_d, df.loc[pullback_buy_d, 'Close'], marker='D',
                color='#00838F', s=90, zorder=6, label='回踩季線(◆)')
    ax1.scatter(exit_sig_d, df.loc[exit_sig_d, 'Close'], marker='v',
                color='#B71C1C', s=130, zorder=7, label='動態出場觸發(▼)')
    ax1.set_ylabel('價格', fontsize=10)
    ax1.legend(loc=LOC, fontsize=8, ncol=5)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('收盤價 + MA20/MA60  ▲強勢突破月線 ◆回踩季線支撐  ▼動態停利停損出場', fontsize=11)

    # ── 圖2：RSI ──
    ax2 = axes[1]
    ax2.plot(dates, df['RSI'], color='#9C27B0', linewidth=1.2, label='RSI(14)')
    ax2.axhline(y=70, color='#F44336', linewidth=1.2, linestyle='--', label='警戒 70')
    ax2.axhline(y=30, color='#4CAF50', linewidth=1.2, linestyle='--', label='超賣 30')
    ax2.axhline(y=50, color='#aaa', linewidth=0.8, linestyle=':')
    ax2.fill_between(dates, 70, df['RSI'].clip(upper=100),
                     where=(df['RSI'] >= 70), alpha=0.08, color='#F44336')
    ax2.fill_between(dates, df['RSI'].clip(lower=0), 30,
                     where=(df['RSI'] <= 30), alpha=0.15, color='#4CAF50')
    rsi_kd_sell_d = df[df['RSI_KD_SELL']].index
    ax2.scatter(rsi_kd_sell_d, df.loc[rsi_kd_sell_d, 'RSI'], marker='v',
                color='#B71C1C', s=80, zorder=6, label='RSI破70+KD死叉')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('RSI', fontsize=9)
    ax2.legend(loc=LOC, fontsize=8, ncol=4)
    ax2.set_title('RSI (14日)  警戒70（跌破70+KD死叉=賣）/ 超賣<=30', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ── 圖3：MACD ──
    ax3 = axes[2]
    osc_colors = ['#F44336' if v >= 0 else '#4CAF50' for v in df['OSC']]
    ax3.bar(dates, df['OSC'],    color=osc_colors, width=1, alpha=0.7, label='OSC 柱狀')
    ax3.plot(dates, df['MACD'],  color='#2196F3', linewidth=1.2, label='MACD')
    ax3.plot(dates, df['SIGNAL'],color='#FF9800', linewidth=1.2, label='Signal')
    ax3.axhline(y=0, color='#555', linewidth=0.8)
    macd_buy_d  = df[df['MACD_BUY']].index
    macd_sell_d = df[df['MACD_SELL']].index
    ax3.scatter(macd_buy_d,  df.loc[macd_buy_d,  'MACD'], marker='^',
                color='#4CAF50', s=70, zorder=5, label='金叉')
    ax3.scatter(macd_sell_d, df.loc[macd_sell_d, 'MACD'], marker='v',
                color='#F44336', s=70, zorder=5, label='死叉')
    osc_shrink_d = df[df['OSC_SHRINK']].index
    ax3.scatter(osc_shrink_d, df.loc[osc_shrink_d, 'OSC'], marker='o',
                color='#FF9800', s=30, zorder=4, alpha=0.7, label='OSC水下收斂')
    ax3.legend(loc=LOC, fontsize=8, ncol=6)
    ax3.set_ylabel('MACD', fontsize=9)
    ax3.set_title('MACD (12/26/9)  ▲金叉 ▼死叉', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ── 圖4：KD ──
    ax4 = axes[3]
    ax4.plot(dates, df['K'], color='#2196F3', linewidth=1.2, label='K')
    ax4.plot(dates, df['D'], color='#FF9800', linewidth=1.2, label='D')
    ax4.axhline(y=80, color='#F44336', linewidth=1.2, linestyle='--', label='超買 80')
    ax4.axhline(y=20, color='#4CAF50', linewidth=1.2, linestyle='--', label='超賣 20')
    ax4.fill_between(dates, 80, 100, alpha=0.06, color='#F44336')
    ax4.fill_between(dates, 0,  20,  alpha=0.06, color='#4CAF50')
    kd_buy_d  = df[df['KD_BUY']].index
    kd_sell_d = df[df['KD_SELL']].index
    ax4.scatter(kd_buy_d,  df.loc[kd_buy_d,  'K'], marker='^', color='#4CAF50', s=60, zorder=5, label='KD金叉')
    ax4.scatter(kd_sell_d, df.loc[kd_sell_d, 'K'], marker='v', color='#F44336', s=60, zorder=5, label='KD死叉')
    ax4.set_ylim(0, 100)
    ax4.legend(loc=LOC, fontsize=8, ncol=3)
    ax4.set_ylabel('KD', fontsize=9)
    ax4.set_title('KD 隨機指標 (9日)  ▲金叉(K<50) ▼死叉(K>50)', fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ── 圖5：成交量 ──
    ax5 = axes[4]
    vol_colors = ['#F44336' if c >= o else '#4CAF50'
                  for c, o in zip(df['Close'], df['Open'])]
    ax5.bar(dates, df['Volume'], color=vol_colors, width=1, alpha=0.7)
    ax5.set_ylabel('成交量', fontsize=9)
    ax5.set_title('成交量', fontsize=10)
    ax5.grid(True, alpha=0.3)

    # 格式化 x 軸
    for ax in axes:
        ax.set_xlim(dates[0], dates[-1])
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{raw_ticker}_technical_analysis.png", dpi=150, bbox_inches='tight')
    print(f"📊 圖表已儲存：{raw_ticker}_technical_analysis.png")
    plt.show()


# ==========================================
# ★ 7. 依序分析每一檔輸入的股票代號
# ==========================================
for idx, raw_ticker in enumerate(raw_tickers, start=1):
    print(f"\n{'#' * 55}\n# ({idx}/{len(raw_tickers)}) 分析代號：{raw_ticker}\n{'#' * 55}")
    analyze_stock(raw_ticker, period)

