"""
临床试验数据分析项目 - 数据探索阶段（NSCLC适配版）
字段名完全匹配 ClinicalTrials.gov 下载格式
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ============================================
# 配置设置
# ============================================

sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# 创建输出目录
OUTPUT_DIR = 'outputs'
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/reports', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/data', exist_ok=True)

# 医学配色方案
MEDICAL_PALETTE = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'neutral': '#6C757D'
}


# ============================================
# 1. 数据加载与验证
# ============================================

def load_and_validate_data(filepath):
    """加载数据并进行基础验证"""
    print("=" * 70)
    print("临床试验数据探索 - NSCLC分析")
    print("=" * 70)

    df = pd.read_csv(filepath)

    print(f"\n📊 数据集概览")
    print(f"   记录数: {len(df):,}")
    print(f"   字段数: {len(df.columns)}")
    print(f"   内存占用: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

    # NCT Number唯一性检查
    unique_nct = df['NCT Number'].nunique()
    print(f"\n🔍 NCT Number唯一性: {unique_nct}/{len(df)} ", end="")
    if unique_nct == len(df):
        print("✅ 通过")
    else:
        print(f"⚠️ 发现 {len(df) - unique_nct} 条重复")
        df = df.drop_duplicates(subset=['NCT Number'], keep='first')
        print(f"   去重后: {len(df):,}")

    return df


# ============================================
# 2. 缺失值深度分析
# ============================================

def analyze_missing_values(df):
    """深度缺失值分析"""
    print("\n" + "=" * 70)
    print("缺失值分析")
    print("=" * 70)

    missing = df.isnull().sum()
    missing_pct = 100 * missing / len(df)

    missing_df = pd.DataFrame({
        '字段类型': df.dtypes,
        '非空数量': df.count(),
        '缺失数量': missing,
        '缺失百分比': missing_pct.round(2)
    })

    # 标记重要性
    high_importance = ['Phases', 'Study Status', 'Start Date', 'Enrollment', 'Sponsor']
    medium_importance = ['Study Type', 'Funder Type', 'Conditions', 'Interventions']

    def get_importance(col):
        if col in high_importance:
            return '高'
        elif col in medium_importance:
            return '中'
        else:
            return '低'

    missing_df['重要性'] = missing_df.index.map(get_importance)

    # 筛选并排序
    missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失百分比', ascending=False)

    print(f"\n发现 {len(missing_df)} 个字段存在缺失值")
    print(missing_df.to_string())

    # 关键字段警告
    critical_missing = missing_df[missing_df['重要性'] == '高']['缺失百分比']
    if not critical_missing.empty:
        print(f"\n⚠️ 关键字段缺失:")
        for field, pct in critical_missing.items():
            print(f"   - {field}: {pct:.1f}%")

    # 保存报告
    missing_df.to_csv(f'{OUTPUT_DIR}/reports/missing_values_report.csv')

    # 可视化
    plt.figure(figsize=(12, 8))
    missing_matrix = df.isnull().astype(int)
    sns.heatmap(missing_matrix.iloc[:min(100, len(df))],
                cbar=True, yticklabels=False,
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Missing (1=Yes, 0=No)'})
    plt.title('缺失值分布热力图（前100条）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/missing_values_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✅ 缺失值热力图已保存")

    return missing_df


# ============================================
# 3. 核心字段分布分析
# ============================================

def analyze_study_type(df):
    """研究类型分析"""
    print("\n" + "=" * 70)
    print("研究类型分布 (Study Type)")
    print("=" * 70)

    if 'Study Type' not in df.columns:
        print("⚠️ Study Type 字段不存在")
        return None

    value_counts = df['Study Type'].value_counts()
    value_pct = (100 * value_counts / len(df)).round(2)

    result_df = pd.DataFrame({'数量': value_counts, '百分比': value_pct})
    print(result_df.to_string())

    # 可视化
    plt.figure(figsize=(10, 6))
    colors = [MEDICAL_PALETTE['primary'], MEDICAL_PALETTE['secondary']]
    ax = value_counts.plot(kind='bar', color=colors[:len(value_counts)])
    plt.title('研究类型分布', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('')
    plt.ylabel('数量', fontsize=12)
    plt.xticks(rotation=0)

    for i, (idx, v) in enumerate(value_counts.items()):
        ax.text(i, v + max(value_counts) * 0.01, f'{v}\n({value_pct[idx]}%)',
                ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/study_type_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 研究类型图已保存")

    return result_df


def analyze_phases(df):
    """试验阶段分析（核心！）"""
    print("\n" + "=" * 70)
    print("试验阶段分布 (Phases)")
    print("=" * 70)

    if 'Phases' not in df.columns:
        print("⚠️ Phases 字段不存在")
        return None

    # 清洗Phases（处理多值如 "Phase 1|Phase 2"）
    def clean_phase(phase):
        if pd.isna(phase):
            return 'Not Specified'
        phase = str(phase)
        # 提取主要阶段（取第一个）
        if '|' in phase:
            return phase.split('|')[0].strip()
        return phase.strip()

    df['Phases_Clean'] = df['Phases'].apply(clean_phase)

    value_counts = df['Phases_Clean'].value_counts()
    value_pct = (100 * value_counts / len(df)).round(2)

    result_df = pd.DataFrame({'数量': value_counts, '百分比': value_pct})
    print(result_df.to_string())

    # 可视化
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(value_counts)))
    ax = value_counts.plot(kind='barh', color=colors)
    plt.title('试验阶段分布（NSCLC）', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('数量', fontsize=12)
    plt.ylabel('')

    for i, (idx, v) in enumerate(value_counts.items()):
        ax.text(v + max(value_counts) * 0.01, i, f'{v} ({value_pct[idx]}%)',
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/phase_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 试验阶段图已保存")

    return result_df


def analyze_status(df):
    """试验状态分析（核心！类似Survived）"""
    print("\n" + "=" * 70)
    print("试验状态分布 (Study Status)")
    print("=" * 70)

    if 'Study Status' not in df.columns:
        print("⚠️ Study Status 字段不存在")
        return None

    value_counts = df['Study Status'].value_counts()
    value_pct = (100 * value_counts / len(df)).round(2)

    result_df = pd.DataFrame({'数量': value_counts, '百分比': value_pct})
    print(result_df.to_string())

    # 定义结果分类（类似Titanic的Survived 0/1）
    def classify_outcome(status):
        if pd.isna(status):
            return 'Unknown'
        status = str(status).upper()
        if 'COMPLETED' in status:
            return 'Completed'
        elif any(x in status for x in ['TERMINATED', 'SUSPENDED', 'WITHDRAWN']):
            return 'Failed'
        elif any(x in status for x in ['RECRUITING', 'ACTIVE', 'NOT YET RECRUITING', 'ENROLLING']):
            return 'Ongoing'
        else:
            return 'Other'

    df['Outcome'] = df['Study Status'].apply(classify_outcome)
    outcome_counts = df['Outcome'].value_counts()

    print(f"\n【试验结果分类】（核心指标，类似Titanic生还率）")
    for outcome, count in outcome_counts.items():
        pct = count / len(df) * 100
        print(f"   {outcome}: {count} ({pct:.1f}%)")

    # 可视化
    plt.figure(figsize=(10, 6))
    color_map = {
        'Completed': MEDICAL_PALETTE['success'],
        'Failed': MEDICAL_PALETTE['danger'],
        'Ongoing': MEDICAL_PALETTE['warning'],
        'Other': MEDICAL_PALETTE['neutral']
    }
    colors = [color_map.get(x, MEDICAL_PALETTE['neutral']) for x in value_counts.index]

    ax = value_counts.plot(kind='bar', color=colors)
    plt.title('试验状态分布（原始）', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('')
    plt.ylabel('数量', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    for i, (idx, v) in enumerate(value_counts.items()):
        ax.text(i, v + max(value_counts) * 0.01, f'{v}\n({value_pct[idx]}%)',
                ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/status_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 试验状态图已保存")

    return result_df, outcome_counts


# ============================================
# 4. 时间序列分析
# ============================================

def analyze_temporal_trends(df):
    """时间趋势分析"""
    print("\n" + "=" * 70)
    print("时间趋势分析")
    print("=" * 70)

    if 'Start Date' not in df.columns:
        print("⚠️ Start Date 字段不存在")
        return None

    # 解析日期
    df['StartDate_Parsed'] = pd.to_datetime(df['Start Date'], errors='coerce')
    df['StartYear'] = df['StartDate_Parsed'].dt.year
    df['StartMonth'] = df['StartDate_Parsed'].dt.month

    # 过滤合理年份
    valid_mask = (df['StartYear'] >= 2015) & (df['StartYear'] <= 2025)
    df_valid = df[valid_mask].copy()

    year_counts = df_valid['StartYear'].value_counts().sort_index()

    print(f"\n年度试验启动数量（2015-2025）:")
    print(year_counts.to_string())

    # 可视化
    plt.figure(figsize=(14, 6))
    plt.plot(year_counts.index, year_counts.values,
             marker='o', linewidth=2.5, markersize=8,
             color=MEDICAL_PALETTE['primary'])
    plt.fill_between(year_counts.index, year_counts.values, alpha=0.3,
                     color=MEDICAL_PALETTE['primary'])

    # COVID-19标注
    plt.axvspan(2020, 2021, alpha=0.15, color='red', label='COVID-19 Pandemic')

    plt.title('NSCLC临床试验年度启动趋势（2015-2025）', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('试验数量', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/yearly_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 年度趋势图已保存")

    # 保存时间特征
    df[['NCT Number', 'StartDate_Parsed', 'StartYear', 'StartMonth']].to_csv(
        f'{OUTPUT_DIR}/data/temporal_features.csv', index=False)

    return year_counts


# ============================================
# 5. 交叉分析（Phases × Outcome）
# ============================================

def analyze_cross_tab(df):
    """Phase与Outcome交叉分析"""
    print("\n" + "=" * 70)
    print("Phase × Outcome 交叉分析")
    print("=" * 70)

    if 'Phases_Clean' not in df.columns or 'Outcome' not in df.columns:
        print("⚠️ 缺少必要字段，请先运行analyze_phases和analyze_status")
        return None

    # 交叉表
    cross_tab = pd.crosstab(df['Phases_Clean'], df['Outcome'], margins=True)
    print("\n数量交叉表:")
    print(cross_tab)

    # 百分比（按Phase）
    cross_pct = pd.crosstab(df['Phases_Clean'], df['Outcome'],
                            normalize='index') * 100
    print("\n百分比交叉表（按Phase）:")
    print(cross_pct.round(2))

    # 热力图
    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_pct, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'Percentage (%)'})
    plt.title('试验阶段 × 结果分布热力图', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('试验结果', fontsize=12)
    plt.ylabel('试验阶段', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/phase_outcome_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 交叉分析热力图已保存")

    return cross_tab, cross_pct


# ============================================
# 6. 生成探索报告
# ============================================

def generate_report(df, missing_df, temporal_data, outcome_counts):
    """生成Markdown报告"""
    report_path = f'{OUTPUT_DIR}/reports/data_exploration_report.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# NSCLC临床试验数据探索报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        f.write(f"## 1. 数据概览\n\n")
        f.write(f"- **总记录数**: {len(df):,}\n")
        f.write(f"- **字段数**: {len(df.columns)}\n")
        f.write(f"- **NCT Number唯一性**: ✅ 通过\n\n")

        f.write(f"## 2. 关键发现\n\n")
        if outcome_counts is not None:
            f.write(f"### 试验成功率（核心指标）\n\n")
            completed_pct = outcome_counts.get('Completed', 0) / len(df) * 100
            failed_pct = outcome_counts.get('Failed', 0) / len(df) * 100
            ongoing_pct = outcome_counts.get('Ongoing', 0) / len(df) * 100
            f.write(f"- **Completed（成功）**: {completed_pct:.1f}%\n")
            f.write(f"- **Failed（失败）**: {failed_pct:.1f}%\n")
            f.write(f"- **Ongoing（进行中）**: {ongoing_pct:.1f}%\n\n")

        if 'Phases_Clean' in df.columns:
            f.write(f"### 试验阶段分布\n\n")
            phase_dist = df['Phases_Clean'].value_counts()
            for phase, count in phase_dist.head(5).items():
                pct = count / len(df) * 100
                f.write(f"- {phase}: {count} ({pct:.1f}%)\n")
            f.write(f"\n")

        f.write(f"## 3. 缺失值情况\n\n")
        if missing_df is not None and not missing_df.empty:
            high_missing = missing_df[missing_df['重要性'] == '高']
            if not high_missing.empty:
                f.write(f"**关键字段缺失**:\n\n")
                for field, row in high_missing.iterrows():
                    f.write(f"- {field}: {row['缺失百分比']:.1f}%\n")
            else:
                f.write(f"关键字段无缺失 ✅\n")
        f.write(f"\n")

        f.write(f"## 4. 输出文件\n\n")
        f.write(f"- 图表: `{OUTPUT_DIR}/figures/` (6张)\n")
        f.write(f"- 报告: `{OUTPUT_DIR}/reports/` \n")
        f.write(f"- 数据: `{OUTPUT_DIR}/data/`\n")

    print(f"\n✅ 报告已保存: {report_path}")


# ============================================
# 主执行流程
# ============================================

def main():
    """主执行函数"""
    # 1. 加载
    df = load_and_validate_data('ctg-studies.csv')

    # 2. 缺失值
    missing_df = analyze_missing_values(df)

    # 3. 核心字段
    analyze_study_type(df)
    analyze_phases(df)
    _, outcome_counts = analyze_status(df)

    # 4. 时间趋势
    temporal_data = analyze_temporal_trends(df)

    # 5. 交叉分析
    analyze_cross_tab(df)

    # 6. 报告
    generate_report(df, missing_df, temporal_data, outcome_counts)

    print(f"\n{'=' * 70}")
    print("数据探索阶段完成！")
    print(f"{'=' * 70}")
    print(f"关键指标（类似Titanic生还率）:")
    if outcome_counts is not None:
        print(f"  - Completed率: {outcome_counts.get('Completed', 0) / len(df) * 100:.1f}%")
        print(f"  - Failed率: {outcome_counts.get('Failed', 0) / len(df) * 100:.1f}%")
    print(f"\n下一步: 运行数据清洗脚本 -> SQL窗口函数分析")


if __name__ == "__main__":
    main()
