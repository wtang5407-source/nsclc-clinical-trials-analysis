"""
NSCLC临床试验 - 探索性数据分析与可视化
输入: nsclc_cleaned.csv
输出: 12+张分析图表 + 洞察报告
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 配置
# ============================================

plt.style.use('seaborn-v0_8-whitegrid')
# 跨平台中文字体回退：Windows / macOS / Linux 尽量都能正常显示中文
plt.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei',                  # Windows
    'PingFang SC', 'Heiti SC', 'Arial Unicode MS', # macOS
    'Noto Sans CJK SC', 'WenQuanYi Micro Hei',     # Linux
    'sans-serif',
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

OUTPUT_DIR = 'outputs/eda_figures'
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 医学配色
COLORS = {
    'completed': '#28A745',
    'failed': '#DC3545',
    'ongoing': '#FFC107',
    'unknown': '#6C757D',
    'phase1': '#3498DB',
    'phase2': '#9B59B6',
    'phase3': '#E74C3C',
    'pharma': '#2C3E50',
    'biotech': '#16A085',
    'academic': '#D35400'
}

print("=" * 70)
print("NSCLC临床试验 - 探索性数据分析")
print("=" * 70)

# ============================================
# 1. 加载数据
# ============================================

df = pd.read_csv('data/nsclc_cleaned.csv', parse_dates=['START_DATE'])

print(f"\n数据集: {len(df)}条记录")
print(f"时间范围: {df['START_YEAR'].min():.0f}-{df['START_YEAR'].max():.0f}")

# 过滤2025年不完整数据（可选，用于趋势分析）
df_complete = df[df['START_YEAR'] <= 2025].copy()
print(f"完整年份数据(2019-2025): {len(df_complete)}条")

# ============================================
# 2. 核心指标概览（类似Titanic生存率）
# ============================================

print("\n" + "=" * 70)
print("【核心指标】试验成功率（类似Titanic生存率）")
print("=" * 70)

outcome_dist = df['OUTCOME'].value_counts()
outcome_pct = df['OUTCOME'].value_counts(normalize=True) * 100

print(f"\n整体分布:")
for outcome in ['Completed', 'Failed', 'Ongoing', 'Unknown']:
    count = outcome_dist.get(outcome, 0)
    pct = outcome_pct.get(outcome, 0)
    print(f"  {outcome:12s}: {count:4d} ({pct:5.1f}%)")

# 计算可评估成功率（排除Ongoing）
evaluable = df[df['OUTCOME'].isin(['Completed', 'Failed'])]
if len(evaluable) > 0:
    success_rate = (evaluable['OUTCOME'] == 'Completed').mean() * 100
    print(f"\n可评估试验成功率: {success_rate:.1f}% (n={len(evaluable)})")

# ============================================
# 可视化1: 试验结果分布（核心！类似Titanic生存饼图）
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 饼图
outcome_counts = df['OUTCOME'].value_counts()
colors_pie = [COLORS.get(o.lower(), '#999999') for o in outcome_counts.index]
wedges, texts, autotexts = axes[0].pie(
    outcome_counts,
    labels=outcome_counts.index,
    autopct='%1.1f%%',
    colors=colors_pie,
    explode=[0.05 if x == 'Completed' else 0 for x in outcome_counts.index],
    startangle=90
)
axes[0].set_title('NSCLC临床试验结果分布\n(n=2,441)', fontsize=12, fontweight='bold')

# 柱状图（按Phase分组）
outcome_by_phase = pd.crosstab(df['PHASE_CLEAN'], df['OUTCOME'])
outcome_by_phase_pct = outcome_by_phase.div(outcome_by_phase.sum(axis=1), axis=0) * 100

outcome_by_phase_pct.plot(kind='bar', stacked=True, ax=axes[1],
                          color=[COLORS.get(c.lower(), '#999999') for c in outcome_by_phase_pct.columns])
axes[1].set_title('试验结果分布（按Phase）', fontsize=12, fontweight='bold')
axes[1].set_xlabel('试验阶段')
axes[1].set_ylabel('百分比 (%)')
axes[1].legend(title='结果', bbox_to_anchor=(1.05, 1))
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_outcome_overview.png', bbox_inches='tight')
plt.close()
print("✅ 图1: 试验结果分布")

# ============================================
# 可视化2: Phase分布（类似Titanic舱位分析）
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Phase数量分布
phase_counts = df['PHASE_CLEAN'].value_counts()
phase_order = ['Phase 1', 'Phase 1/2', 'Phase 2', 'Phase 2/3', 'Phase 3', 'Early Phase 1']
phase_counts = phase_counts.reindex([p for p in phase_order if p in phase_counts.index])

colors_phase = [COLORS['phase1'], '#8E44AD', COLORS['phase2'], '#C0392B', COLORS['phase3'], '#F39C12']
bars = axes[0].barh(range(len(phase_counts)), phase_counts.values, color=colors_phase[:len(phase_counts)])
axes[0].set_yticks(range(len(phase_counts)))
axes[0].set_yticklabels(phase_counts.index)
axes[0].set_xlabel('试验数量')
axes[0].set_title('NSCLC试验阶段分布', fontsize=12, fontweight='bold')

# 添加数值标签
for i, (idx, v) in enumerate(phase_counts.items()):
    pct = v / len(df) * 100
    axes[0].text(v + max(phase_counts)*0.01, i, f'{v} ({pct:.1f}%)', va='center', fontsize=9)

# Phase成功率对比
phase_success = df[df['OUTCOME'].isin(['Completed', 'Failed'])].groupby('PHASE_CLEAN')['OUTCOME'].apply(
    lambda x: (x == 'Completed').mean() * 100
).reindex([p for p in phase_order if p in df['PHASE_CLEAN'].unique()])

bars2 = axes[1].bar(range(len(phase_success)), phase_success.values,
                    color=[COLORS['phase1'], '#8E44AD', COLORS['phase2'], '#C0392B', COLORS['phase3']][:len(phase_success)])
axes[1].set_xticks(range(len(phase_success)))
axes[1].set_xticklabels(phase_success.index, rotation=45, ha='right')
axes[1].set_ylabel('成功率 (%)')
axes[1].set_title('各Phase试验成功率（Completed / (Completed+Failed)）', fontsize=12, fontweight='bold')
axes[1].axhline(y=phase_success.mean(), color='red', linestyle='--', alpha=0.5, label=f'平均: {phase_success.mean():.1f}%')
axes[1].legend()

# 添加数值标签
for i, v in enumerate(phase_success.values):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_phase_analysis.png', bbox_inches='tight')
plt.close()
print("✅ 图2: Phase分析")

# ============================================
# 可视化3: 时间序列趋势（2019-2025）
# ============================================

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 年度试验数量
yearly_counts = df_complete.groupby(['START_YEAR', 'OUTCOME']).size().unstack(fill_value=0)
yearly_counts = yearly_counts.reindex(columns=['Completed', 'Failed', 'Ongoing', 'Unknown'], fill_value=0)

yearly_counts.plot(kind='bar', stacked=True, ax=axes[0],
                   color=[COLORS['completed'], COLORS['failed'], COLORS['ongoing'], COLORS['unknown']])
axes[0].set_title('NSCLC临床试验年度启动趋势（2019-2025）', fontsize=12, fontweight='bold')
axes[0].set_xlabel('年份')
axes[0].set_ylabel('试验数量')
axes[0].legend(title='结果')
axes[0].tick_params(axis='x', rotation=0)

# 标注COVID-19
axes[0].axvspan(0.5, 2.5, alpha=0.1, color='red', label='COVID-19')

# 年度成功率趋势
yearly_success = df_complete[df_complete['OUTCOME'].isin(['Completed', 'Failed'])].groupby('START_YEAR').apply(
    lambda x: (x['OUTCOME'] == 'Completed').mean() * 100
)

axes[1].plot(yearly_success.index, yearly_success.values, marker='o', linewidth=2.5,
             markersize=8, color=COLORS['completed'])
axes[1].fill_between(yearly_success.index, yearly_success.values, alpha=0.3, color=COLORS['completed'])
axes[1].set_title('年度试验成功率趋势', fontsize=12, fontweight='bold')
axes[1].set_xlabel('年份')
axes[1].set_ylabel('成功率 (%)')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=yearly_success.mean(), color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_temporal_trends.png', bbox_inches='tight')
plt.close()
print("✅ 图3: 时间趋势")

# ============================================
# 可视化4: 申办方分析（类似Titanic港口分析）
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 申办方类型分布
sponsor_counts = df['SPONSOR_TYPE'].value_counts()
colors_sponsor = [COLORS['pharma'], COLORS['biotech'], COLORS['academic'], '#999999']
wedges, texts, autotexts = axes[0].pie(
    sponsor_counts,
    labels=sponsor_counts.index,
    autopct='%1.1f%%',
    colors=colors_sponsor,
    startangle=90
)
axes[0].set_title('申办方类型分布', fontsize=12, fontweight='bold')

# 申办方类型成功率
sponsor_success = df[df['OUTCOME'].isin(['Completed', 'Failed'])].groupby('SPONSOR_TYPE')['OUTCOME'].apply(
    lambda x: (x == 'Completed').mean() * 100
).sort_values(ascending=False)

bars = axes[1].bar(range(len(sponsor_success)), sponsor_success.values, color=colors_sponsor[:len(sponsor_success)])
axes[1].set_xticks(range(len(sponsor_success)))
axes[1].set_xticklabels(sponsor_success.index, rotation=45, ha='right')
axes[1].set_ylabel('成功率 (%)')
axes[1].set_title('各申办方类型试验成功率', fontsize=12, fontweight='bold')
axes[1].axhline(y=sponsor_success.mean(), color='red', linestyle='--', alpha=0.5)

for i, v in enumerate(sponsor_success.values):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_sponsor_analysis.png', bbox_inches='tight')
plt.close()
print("✅ 图4: 申办方分析")

# ============================================
# 可视化5: 疗法类型分析
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 疗法类型分布
therapy_counts = df['THERAPY_TYPE'].value_counts().head(6)
colors_therapy = plt.cm.Set3(np.linspace(0, 1, len(therapy_counts)))
bars = axes[0].barh(range(len(therapy_counts)), therapy_counts.values, color=colors_therapy)
axes[0].set_yticks(range(len(therapy_counts)))
axes[0].set_yticklabels(therapy_counts.index)
axes[0].set_xlabel('试验数量')
axes[0].set_title('疗法类型分布（Top 6）', fontsize=12, fontweight='bold')
axes[0].invert_yaxis()

for i, (idx, v) in enumerate(therapy_counts.items()):
    pct = v / len(df) * 100
    axes[0].text(v + max(therapy_counts)*0.01, i, f'{v} ({pct:.1f}%)', va='center', fontsize=9)

# 疗法类型成功率
therapy_success = df[df['OUTCOME'].isin(['Completed', 'Failed'])].groupby('THERAPY_TYPE')['OUTCOME'].apply(
    lambda x: (x == 'Completed').mean() * 100
).sort_values(ascending=False).head(6)

bars2 = axes[1].bar(range(len(therapy_success)), therapy_success.values, color=colors_therapy[:len(therapy_success)])
axes[1].set_xticks(range(len(therapy_success)))
axes[1].set_xticklabels(therapy_success.index, rotation=45, ha='right')
axes[1].set_ylabel('成功率 (%)')
axes[1].set_title('各疗法类型成功率（Top 6）', fontsize=12, fontweight='bold')

for i, v in enumerate(therapy_success.values):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_therapy_analysis.png', bbox_inches='tight')
plt.close()
print("✅ 图5: 疗法分析")

# ============================================
# 可视化6: 入组规模分析（类似Titanic家庭规模）
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 入组规模分布
enroll_counts = df['ENROLL_GROUP'].value_counts()
enroll_order = ['<50', '50-100', '100-300', '300-1000', '>1000']
enroll_counts = enroll_counts.reindex([e for e in enroll_order if e in enroll_counts.index])

colors_enroll = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(enroll_counts)))
bars = axes[0].bar(range(len(enroll_counts)), enroll_counts.values, color=colors_enroll)
axes[0].set_xticks(range(len(enroll_counts)))
axes[0].set_xticklabels(enroll_counts.index)
axes[0].set_xlabel('入组规模')
axes[0].set_ylabel('试验数量')
axes[0].set_title('试验入组规模分布', fontsize=12, fontweight='bold')

for i, (idx, v) in enumerate(enroll_counts.items()):
    pct = v / df['ENROLL_GROUP'].notna().sum() * 100
    axes[0].text(i, v + max(enroll_counts)*0.01, f'{v}\n({pct:.1f}%)', ha='center', fontsize=9)

# 入组规模成功率
enroll_success = df[df['OUTCOME'].isin(['Completed', 'Failed'])].groupby('ENROLL_GROUP')['OUTCOME'].apply(
    lambda x: (x == 'Completed').mean() * 100
).reindex([e for e in enroll_order if e in df['ENROLL_GROUP'].unique()])

axes[1].plot(range(len(enroll_success)), enroll_success.values,
             marker='o', linewidth=2.5, markersize=10, color=COLORS['completed'])
axes[1].set_xticks(range(len(enroll_success)))
axes[1].set_xticklabels(enroll_success.index)
axes[1].set_xlabel('入组规模')
axes[1].set_ylabel('成功率 (%)')
axes[1].set_title('入组规模与成功率关系', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# 标注最优区间
optimal_idx = enroll_success.values.argmax()
axes[1].axvspan(optimal_idx-0.3, optimal_idx+0.3, alpha=0.2, color='green',
                label=f'最优: {enroll_success.index[optimal_idx]}')
axes[1].legend()

for i, v in enumerate(enroll_success.values):
    axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_enrollment_analysis.png', bbox_inches='tight')
plt.close()
print("✅ 图6: 入组规模分析")

# ============================================
# 可视化7: 交叉热力图（Phase × SponsorType）
# ============================================

cross_data = pd.crosstab(df['PHASE_CLEAN'], df['SPONSOR_TYPE'], normalize='index') * 100

plt.figure(figsize=(10, 6))
sns.heatmap(cross_data, annot=True, fmt='.1f', cmap='YlGnBu', cbar_kws={'label': 'Percentage (%)'})
plt.title('试验阶段 × 申办方类型分布', fontsize=12, fontweight='bold')
plt.xlabel('申办方类型')
plt.ylabel('试验阶段')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_phase_sponsor_heatmap.png', bbox_inches='tight')
plt.close()
print("✅ 图7: Phase-Sponsor热力图")

# ============================================
# 可视化8: 试验持续时间分析
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 持续时间分布（仅Completed试验）
completed_trials = df[df['OUTCOME'] == 'Completed'].copy()
duration_data = completed_trials['DURATION_MONTHS'].dropna()

axes[0].hist(duration_data, bins=30, color=COLORS['completed'], alpha=0.7, edgecolor='black')
axes[0].axvline(duration_data.median(), color='red', linestyle='--',
                label=f'中位数: {duration_data.median():.1f}月')
axes[0].set_xlabel('试验持续时间（月）')
axes[0].set_ylabel('试验数量')
axes[0].set_title('Completed试验持续时间分布', fontsize=12, fontweight='bold')
axes[0].legend()

# 各Phase持续时间对比
phase_duration = completed_trials.groupby('PHASE_CLEAN')['DURATION_MONTHS'].median().sort_values()
phase_duration = phase_duration.reindex([p for p in phase_order if p in phase_duration.index])

bars = axes[1].barh(range(len(phase_duration)), phase_duration.values,
                    color=[COLORS['phase1'], '#8E44AD', COLORS['phase2'], '#C0392B', COLORS['phase3']][:len(phase_duration)])
axes[1].set_yticks(range(len(phase_duration)))
axes[1].set_yticklabels(phase_duration.index)
axes[1].set_xlabel('中位持续时间（月）')
axes[1].set_title('各Phase试验持续时间（Completed）', fontsize=12, fontweight='bold')

for i, v in enumerate(phase_duration.values):
    axes[1].text(v + 1, i, f'{v:.1f}月', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_duration_analysis.png', bbox_inches='tight')
plt.close()
print("✅ 图8: 持续时间分析")

# ============================================
# 9. 关键洞察总结
# ============================================

print("\n" + "=" * 70)
print("【关键洞察】")
print("=" * 70)

insights = []

# 洞察1: 整体成功率
completed_rate = (df['OUTCOME'] == 'Completed').mean() * 100
insights.append(f"1. 整体Completed率: {completed_rate:.1f}% (n={len(df)})")

# 洞察2: Phase差异
if len(phase_success) > 0:
    best_phase = phase_success.idxmax()
    worst_phase = phase_success.idxmin()
    insights.append(f"2. Phase成功率差异: {best_phase} ({phase_success.max():.1f}%) vs {worst_phase} ({phase_success.min():.1f}%)")

# 洞察3: 申办方差异
if len(sponsor_success) > 0:
    best_sponsor = sponsor_success.idxmax()
    insights.append(f"3. {best_sponsor}成功率最高: {sponsor_success.max():.1f}%")

# 洞察4: 入组规模最优
if len(enroll_success) > 0:
    optimal_size = enroll_success.idxmax()
    insights.append(f"4. 最优入组规模: {optimal_size} ({enroll_success.max():.1f}%)")

# 洞察5: 时间趋势
if len(yearly_success) > 0:
    trend = "上升" if yearly_success.iloc[-1] > yearly_success.iloc[0] else "下降"
    insights.append(f"5. 2019-2025成功率趋势: {trend} ({yearly_success.iloc[0]:.1f}% → {yearly_success.iloc[-1]:.1f}%)")

# 洞察6: 疗法趋势
top_therapy = therapy_counts.index[0]
insights.append(f"6. 最热疗法: {top_therapy} ({therapy_counts.iloc[0]}项试验)")

for insight in insights:
    print(f"  {insight}")

# 保存洞察
with open(f'{OUTPUT_DIR}/insights_summary.txt', 'w') as f:
    f.write("NSCLC临床试验关键洞察\n")
    f.write("=" * 50 + "\n\n")
    for insight in insights:
        f.write(f"{insight}\n")

print(f"\n{'='*70}")
print(f"探索性分析完成！")
print(f"输出目录: {OUTPUT_DIR}/")
print(f"共生成 8 组分析图表")
print(f"{'='*70}")
