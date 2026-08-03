import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

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
INDEX_TICKER = "^TWII"          # 台股加權指數
PERIOD        = "5y"            # 回測期間（可改 5y / 10y / max）
BINS = [-np.inf, -9, -8, -7, -6, -5, -4, -3]  # 回檔區間（%），只看跌 3% 以上

# ==========================================
# ★ 2. 下載資料
# ==========================================
print(f"📥 正在下載 {INDEX_TICKER} 資料（{PERIOD}）...")
df = yf.download(INDEX_TICKER, period=PERIOD, progress=False)

if df.empty:
    print("❌ 資料下載失敗，請確認網路連線。")
    exit()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

MONTH_DAYS = 21   # 約一個月交易日

df = df[['Close']].copy()
df['Return_T']  = df['Close'].pct_change() * 100                           # 當天漲跌幅 (%)
df['Return_T1'] = df['Close'].pct_change(periods=MONTH_DAYS).shift(-MONTH_DAYS) * 100  # 隔月漲跌幅 (%)
df = df.dropna()

total_days = len(df)
print(f"✅ 共取得 {total_days} 個交易日資料（{df.index[0].date()} ~ {df.index[-1].date()}）\n")

# ==========================================
# ★ 3. 只保留「當天下跌」的日子
# ==========================================
decline_df = df[df['Return_T'] < 0].copy()

# 依照回檔幅度分組
labels = [f"{BINS[i]}% ~ {BINS[i+1]}%" for i in range(len(BINS)-1)]
labels[-1] = f"< {BINS[-2]}%"   # 最後一組改成「小於 -0.5%」等
# 重新建立正確 labels
labels = []
for i in range(len(BINS) - 1):
    lo = BINS[i]
    hi = BINS[i+1]
    if lo == -np.inf:
        labels.append(f"< {hi:.1f}%")
    else:
        labels.append(f"{lo:.1f}% ~ {hi:.1f}%")

decline_df['Bin'] = pd.cut(decline_df['Return_T'], bins=BINS, labels=labels)
decline_df = decline_df.dropna(subset=['Bin'])

# ==========================================
# ★ 4. 統計每個回檔區間的隔日表現
# ==========================================
results = []
for label in labels:
    group = decline_df[decline_df['Bin'] == label]
    count = len(group)
    if count == 0:
        continue

    up_months    = (group['Return_T1'] > 0).sum()
    up_rate      = up_months / count * 100
    avg_next     = group['Return_T1'].mean()
    median_next  = group['Return_T1'].median()
    avg_decline  = group['Return_T'].mean()

    results.append({
        '回檔區間':    label,
        '樣本數':      count,
        '隔月上漲次數': up_months,
        '隔月上漲率(%)': round(up_rate, 1),
        '隔月平均漲跌(%)': round(avg_next, 2),
        '隔月中位數(%)':  round(median_next, 2),
        '平均回檔幅度(%)': round(avg_decline, 2),
    })

result_df = pd.DataFrame(results)

# ==========================================
# ★ 5. 印出統計結果
# ==========================================
print("=" * 70)
print(f" 📊 台股大盤回檔後 隔月反彈統計（約 {MONTH_DAYS} 個交易日，回測期間：{PERIOD}）")
print("=" * 70)
print(result_df.to_string(index=False))
print()

# 特別標示上漲率 > 55% 的區間
strong = result_df[result_df['隔月上漲率(%)'] > 55]
if not strong.empty:
    print("🌟 隔月上漲機率 > 55% 的回檔區間：")
    for _, row in strong.iterrows():
        print(f"   {row['回檔區間']}  →  隔月上漲率 {row['隔月上漲率(%)']}%，平均漲 {row['隔月平均漲跌(%)']}%（樣本 {row['樣本數']} 次）")

# ==========================================
# ★ 明細清單：跌 3% 以上的所有日期
# ==========================================
detail_df = decline_df[decline_df['Return_T'] <= -3][
    ['Close', 'Return_T', 'Return_T1']
].copy()
detail_df.columns = ['收盤點', '當日跌幅(%)', '隔月漲跌(%)']
detail_df = detail_df.round(2).sort_index()
detail_df['隔月結果'] = detail_df['隔月漲跌(%)'].apply(
    lambda x: f"↑ +{x:.2f}%" if x > 0 else f"↓ {x:.2f}%"
)
detail_df.index = detail_df.index.strftime('%Y-%m-%d')

print()
print("=" * 65)
print(f" 📅 跌幅 >= 3% 的日期明細（共 {len(detail_df)} 次）")
print("=" * 65)
print(f"{'日期':<12} {'收盤點':>8} {'當日跌幅':>10} {'隔月結果':>14}")
print("-" * 50)
for date_str, row in detail_df.iterrows():
    up_mark = "★" if row['隔月漲跌(%)'] > 0 else "  "
    print(f"{up_mark} {date_str}  {row['收盤點']:>9,.0f}  {row['當日跌幅(%)']:>8.2f}%  {row['隔月結果']:>14}")

up_count   = (detail_df['隔月漲跌(%)'] > 0).sum()
down_count = len(detail_df) - up_count
print("-" * 50)
print(f"  合計：隔月上漲 {up_count} 次（★），下跌 {down_count} 次，上漲率 {up_count/len(detail_df)*100:.1f}%")

# ==========================================
# ★ 6. 視覺化
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'台股大盤（{INDEX_TICKER}）回檔後隔月反彈分析（約 {MONTH_DAYS} 個交易日）\n回測期間：{PERIOD}，共 {total_days} 個交易日',
             fontsize=14, fontweight='bold')

x = range(len(result_df))
tick_labels = result_df['回檔區間'].tolist()

# --- 左圖：隔日上漲率 ---
ax1 = axes[0]
bars = ax1.bar(x, result_df['隔月上漲率(%)'], color=[
    'tomato' if v < 50 else 'steelblue' for v in result_df['隔月上漲率(%)']
], edgecolor='white', linewidth=0.8)
ax1.axhline(50, color='gray', linestyle='--', linewidth=1, label='50% 基準線')
for bar, val in zip(bars, result_df['隔月上漲率(%)']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
ax1.set_xticks(x)
ax1.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=9)
ax1.set_ylabel('隔月上漲機率 (%)')
ax1.set_title('各回檔區間 → 隔月上漲機率')
ax1.legend()
ax1.set_ylim(0, 100)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# --- 右圖：隔日平均漲跌 ---
ax2 = axes[1]
colors2 = ['tomato' if v < 0 else 'steelblue' for v in result_df['隔月平均漲跌(%)']]
bars2 = ax2.bar(x, result_df['隔月平均漲跌(%)'], color=colors2, edgecolor='white', linewidth=0.8)
ax2.axhline(0, color='black', linewidth=1)
for bar, val in zip(bars2, result_df['隔月平均漲跌(%)']):
    offset = 0.1 if val >= 0 else -0.3
    ax2.text(bar.get_x() + bar.get_width()/2, val + offset,
             f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
ax2.set_xticks(x)
ax2.set_xticklabels(tick_labels, rotation=35, ha='right', fontsize=9)
ax2.set_ylabel('隔月平均漲跌 (%)')
ax2.set_title('各回檔區間 → 隔月平均報酬')
ax2.grid(axis='y', linestyle='--', alpha=0.5)

# 在 bar 上標註樣本數
for i, row in result_df.iterrows():
    ax2.text(i, ax2.get_ylim()[0] + 0.05,
             f"n={row['樣本數']}", ha='center', va='bottom', fontsize=8, color='dimgray')

plt.tight_layout()
plt.show()
print("\n✅ 分析完成！")
