"""
NSCLC临床试验数据 - 清洗与预处理
输入: ctg-studies.csv (原始下载)
输出: nsclc_cleaned.csv (清洗后)
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ============================================
# 配置
# ============================================

INPUT_FILE = 'data/ctg-studies.csv'
OUTPUT_FILE = 'data/nsclc_cleaned.csv'

# 医学配色（用于后续可视化）
MEDICAL_PALETTE = {
    'completed': '#28A745',
    'failed': '#DC3545',
    'ongoing': '#FFC107',
    'unknown': '#6C757D'
}

print("=" * 70)
print("NSCLC临床试验数据清洗")
print("=" * 70)

# ============================================
# 1. 加载数据
# ============================================

print("\n【1/7】加载原始数据...")
df = pd.read_csv(INPUT_FILE)
print(f"原始记录: {len(df)}条")
print(f"原始字段: {len(df.columns)}个")

# ============================================
# 2. 核心字段验证与重命名
# ============================================

print("\n【2/7】字段标准化...")

# 创建标准字段名映射（保留原始字段，创建清洗后字段）
df['NCT_ID'] = df['NCT Number']
df['TITLE'] = df['Study Title']
df['STATUS'] = df['Study Status']
df['PHASE_RAW'] = df['Phases']
df['START_DATE_RAW'] = df['Start Date']
df['ENROLL_RAW'] = df['Enrollment']
df['SPONSOR_RAW'] = df['Sponsor']
df['FUNDER'] = df['Funder Type']
df['CONDITIONS'] = df['Conditions']
df['INTERVENTIONS'] = df['Interventions']

# ============================================
# 3. PHASE清洗（核心！）
# ============================================

print("\n【3/7】清洗试验阶段...")


def clean_phase(phase_str):
    """
    清洗Phase字段
    处理: "Phase 1|Phase 2" -> "Phase 1/2", "PHASE1" -> "Phase 1"
    """
    if pd.isna(phase_str) or str(phase_str).strip() == '':
        return 'Not Specified'

    phase_str = str(phase_str).upper().strip()

    # 处理组合阶段
    if '|' in phase_str:
        phases = [p.strip() for p in phase_str.split('|')]
        # 提取数字
        nums = []
        for p in phases:
            match = re.search(r'(\d+)', p)
            if match:
                nums.append(match.group(1))
        if len(nums) >= 2:
            return f"Phase {nums[0]}/{nums[1]}"
        elif len(nums) == 1:
            return f"Phase {nums[0]}"
        else:
            return 'Combined Phase'

    # 处理单一阶段
    if 'EARLY' in phase_str and '1' in phase_str:
        return 'Early Phase 1'
    elif re.search(r'PHASE\s*1[\s/]*2', phase_str) or phase_str == 'PHASE1/PHASE2':
        return 'Phase 1/2'
    elif re.search(r'PHASE\s*2[\s/]*3', phase_str) or phase_str == 'PHASE2/PHASE3':
        return 'Phase 2/3'
    elif '1' in phase_str and '2' not in phase_str and '3' not in phase_str:
        return 'Phase 1'
    elif '2' in phase_str and '1' not in phase_str and '3' not in phase_str:
        return 'Phase 2'
    elif '3' in phase_str and '1' not in phase_str and '2' not in phase_str:
        return 'Phase 3'
    elif '4' in phase_str:
        return 'Phase 4'
    else:
        return 'Not Specified'


df['PHASE_CLEAN'] = df['PHASE_RAW'].apply(clean_phase)

# 验证
print(f"Phase清洗后分布:")
print(df['PHASE_CLEAN'].value_counts().to_string())

# ============================================
# 4. 试验结果分类（核心！类似Survived）
# ============================================

print("\n【4/7】定义试验结果...")


def classify_outcome(status):
    """
    将Study Status分类为: Completed / Failed / Ongoing / Unknown
    """
    if pd.isna(status):
        return 'Unknown'

    status = str(status).upper()

    # 成功完成
    if 'COMPLETED' in status:
        return 'Completed'

    # 失败/终止
    elif any(x in status for x in ['TERMINATED', 'SUSPENDED', 'WITHDRAWN']):
        return 'Failed'

    # 进行中
    elif any(x in status for x in ['RECRUITING', 'ACTIVE', 'ENROLLING', 'NOT YET RECRUITING']):
        return 'Ongoing'

    # 其他/未知
    else:
        return 'Unknown'


df['OUTCOME'] = df['STATUS'].apply(classify_outcome)

# 验证
outcome_dist = df['OUTCOME'].value_counts()
print(f"\n试验结果分布:")
for outcome, count in outcome_dist.items():
    pct = count / len(df) * 100
    print(f"  {outcome}: {count} ({pct:.1f}%)")

# ============================================
# 5. 日期清洗（核心！）
# ============================================

print("\n【5/7】清洗日期字段...")


def parse_date(date_str):
    """
    解析多种日期格式
    """
    if pd.isna(date_str) or str(date_str).strip() == '':
        return pd.NaT

    date_str = str(date_str).strip()

    # 尝试多种格式
    formats = [
        '%Y-%m-%d',  # 2020-01-15
        '%B %Y',  # January 2020
        '%b %Y',  # Jan 2020
        '%Y',  # 2020
        '%m/%d/%Y',  # 01/15/2020
    ]

    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue

    # 最后尝试自动解析
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT


df['START_DATE'] = df['START_DATE_RAW'].apply(parse_date)
df['START_YEAR'] = df['START_DATE'].dt.year
df['START_MONTH'] = df['START_DATE'].dt.month

# 其他日期
df['PRIMARY_COMPLETION'] = df['Primary Completion Date'].apply(parse_date)
df['COMPLETION_DATE'] = df['Completion Date'].apply(parse_date)

# 验证
print(f"\n日期解析结果:")
print(f"  Start Date有效: {df['START_DATE'].notna().sum()}/{len(df)}")
print(f"  年份范围: {df['START_YEAR'].min():.0f} - {df['START_YEAR'].max():.0f}")

# 标记2025年数据为不完整
df['YEAR_COMPLETE'] = df['START_YEAR'].apply(lambda x: 'Yes' if x <= 2024 else 'Partial (2025)')

# ============================================
# 6. Enrollment清洗（核心！）
# ============================================

print("\n【6/7】清洗入组人数...")


def clean_enrollment(val):
    """
    提取Enrollment中的数字
    处理: "100", "approximately 100", "0 (actual: 25)"
    """
    if pd.isna(val):
        return np.nan

    val_str = str(val).upper()

    # 查找所有数字
    numbers = re.findall(r'\d+', val_str)
    if numbers:
        # 取第一个数字
        num = int(numbers[0])
        # 标记特殊值
        if num == 0:
            return np.nan  # 0视为缺失
        return num

    return np.nan


df['ENROLLMENT'] = df['ENROLL_RAW'].apply(clean_enrollment)

# 分箱（类似Titanic的FamilySize）
df['ENROLL_GROUP'] = pd.cut(df['ENROLLMENT'],
                            bins=[0, 50, 100, 300, 1000, float('inf')],
                            labels=['<50', '50-100', '100-300', '300-1000', '>1000'],
                            include_lowest=True)

print(f"\nEnrollment清洗结果:")
print(f"  有效值: {df['ENROLLMENT'].notna().sum()}/{len(df)}")
print(f"  平均值: {df['ENROLLMENT'].mean():.0f}")
print(f"  中位数: {df['ENROLLMENT'].median():.0f}")
print(f"\n分箱分布:")
print(df['ENROLL_GROUP'].value_counts().to_string())

# ============================================
# 7. 申办方分类
# ============================================

print("\n【7/7】分类申办方...")


def classify_sponsor(sponsor):
    """
    分类申办方为: Big Pharma / Biotech / Academic / Other
    """
    if pd.isna(sponsor):
        return 'Unknown'

    sponsor = str(sponsor).upper()

    # Big Pharma名单
    big_pharma = [
        'PFIZER', 'ROCHE', 'NOVARTIS', 'MERCK', 'ASTRAZENECA',
        'BRISTOL MYERS SQUIBB', 'BMS', 'JOHNSON & JOHNSON', 'JANSSEN',
        'ELI LILLY', 'LILLY', 'SANOFI', 'GLAXOSMITHKLINE', 'GSK',
        'ABBVIE', 'TAKEDA', 'BAYER', 'BOEHRINGER INGELHEIM', 'AMGEN',
        'GILEAD', 'REGENERON', 'VERTEX', 'BIOGEN', 'ALEXION'
    ]

    # Academic关键词
    academic = ['UNIVERSITY', 'HOSPITAL', 'MEDICAL CENTER', 'INSTITUTE',
                'COLLEGE', 'SCHOOL OF MEDICINE', 'CANCER CENTER']

    # Biotech关键词（排除Big Pharma后）
    biotech = ['BIOTECH', 'THERAPEUTICS', 'ONCOLOGY', 'BIOSCIENCES',
               'PHARMA', 'LIFE SCIENCES']

    if any(x in sponsor for x in big_pharma):
        return 'Big Pharma'
    elif any(x in sponsor for x in academic):
        return 'Academic'
    elif any(x in sponsor for x in biotech):
        return 'Biotech'
    else:
        return 'Other Industry'


df['SPONSOR_TYPE'] = df['SPONSOR_RAW'].apply(classify_sponsor)

print(f"\n申办方类型分布:")
print(df['SPONSOR_TYPE'].value_counts().to_string())

# ============================================
# 8. 衍生特征工程
# ============================================

print("\n【特征工程】创建衍生字段...")


# 8.1 试验持续时间（月）
def calc_duration(row):
    """计算试验持续时间"""
    start = row['START_DATE']
    end = row['PRIMARY_COMPLETION'] if pd.notna(row['PRIMARY_COMPLETION']) else row['COMPLETION_DATE']

    if pd.notna(start) and pd.notna(end) and end >= start:
        return (end - start).days / 30.44
    return np.nan


df['DURATION_MONTHS'] = df.apply(calc_duration, axis=1)


# 8.2 疗法类型提取
def extract_therapy(intervention):
    """提取主要疗法类型"""
    if pd.isna(intervention):
        return 'Unknown'

    inter = str(intervention).upper()

    if any(x in inter for x in ['PD-1', 'PD-L1', 'CTLA-4', 'IMMUNE', 'CHECKPOINT']):
        return 'Immunotherapy'
    elif any(x in inter for x in ['EGFR', 'ALK', 'ROS1', 'BRAF', 'KRAS', 'TARGETED', 'TYROSINE KINASE']):
        return 'Targeted Therapy'
    elif any(x in inter for x in ['CHEMO', 'PLATINUM', 'TAXANE', 'GEMCITABINE']):
        return 'Chemotherapy'
    elif any(x in inter for x in ['RADIATION', 'RADIOTHERAPY']):
        return 'Radiation'
    elif any(x in inter for x in ['SURGERY', 'RESECTION']):
        return 'Surgery'
    elif any(x in inter for x in ['CAR-T', 'CELL', 'GENE THERAPY']):
        return 'Cell/Gene Therapy'
    else:
        return 'Other/Combination'


df['THERAPY_TYPE'] = df['INTERVENTIONS'].apply(extract_therapy)

print(f"\n疗法类型分布:")
print(df['THERAPY_TYPE'].value_counts().head(6).to_string())

# 8.3 是否披露结果
df['HAS_RESULTS'] = df['Study Results'].apply(
    lambda x: 'Yes' if pd.notna(x) and str(x).upper() == 'HAS RESULTS' else 'No'
)

# ============================================
# 9. 选择输出字段
# ============================================

print("\n【输出准备】整理最终字段...")

output_columns = [
    # 标识
    'NCT_ID', 'TITLE',

    # 核心分类
    'PHASE_CLEAN', 'OUTCOME', 'STATUS',

    # 时间
    'START_DATE', 'START_YEAR', 'START_MONTH', 'YEAR_COMPLETE',
    'PRIMARY_COMPLETION', 'COMPLETION_DATE', 'DURATION_MONTHS',

    # 规模
    'ENROLLMENT', 'ENROLL_GROUP',

    # 申办方
    'SPONSOR_RAW', 'SPONSOR_TYPE', 'FUNDER',

    # 医学信息
    'CONDITIONS', 'INTERVENTIONS', 'THERAPY_TYPE',

    # 其他
    'HAS_RESULTS', 'ACRONYM'
]

# 确保所有字段存在
for col in output_columns:
    if col not in df.columns:
        df[col] = np.nan

df_output = df[output_columns].copy()

# ============================================
# 10. 保存与验证
# ============================================

print("\n" + "=" * 70)
print("清洗完成！")
print("=" * 70)

# 保存
df_output.to_csv(OUTPUT_FILE, index=False)
print(f"\n输出文件: {OUTPUT_FILE}")
print(f"输出记录: {len(df_output)}条")
print(f"输出字段: {len(df_output.columns)}个")

# 验证报告
print(f"\n【数据质量验证】")

print(f"\n1. 核心字段完整性:")
print(f"   NCT_ID: {df_output['NCT_ID'].notna().sum()}/{len(df_output)}")
print(f"   PHASE_CLEAN: {df_output['PHASE_CLEAN'].notna().sum()}/{len(df_output)}")
print(f"   OUTCOME: {df_output['OUTCOME'].notna().sum()}/{len(df_output)}")

print(f"\n2. 关键指标:")
print(f"   Completed率: {(df_output['OUTCOME'] == 'Completed').mean() * 100:.1f}%")
print(f"   Failed率: {(df_output['OUTCOME'] == 'Failed').mean() * 100:.1f}%")
print(f"   Ongoing率: {(df_output['OUTCOME'] == 'Ongoing').mean() * 100:.1f}%")

print(f"\n3. Phase分布:")
for phase in ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 1/2', 'Phase 2/3']:
    count = (df_output['PHASE_CLEAN'] == phase).sum()
    pct = count / len(df_output) * 100
    print(f"   {phase}: {count} ({pct:.1f}%)")

print(f"\n4. 时间范围:")
print(f"   {df_output['START_YEAR'].min():.0f} - {df_output['START_YEAR'].max():.0f}")
print(f"   2025年数据: {(df_output['START_YEAR'] == 2025).sum()}条 (标记为不完整)")

print(f"\n5. 申办方分布:")
for stype in df_output['SPONSOR_TYPE'].value_counts().head(4).index:
    count = (df_output['SPONSOR_TYPE'] == stype).sum()
    pct = count / len(df_output) * 100
    print(f"   {stype}: {count} ({pct:.1f}%)")

print("\n" + "=" * 70)
print("下一步: 运行 03_eda_analysis.py")
print("=" * 70)
