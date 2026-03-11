# Define the data structure, regularization and formatting methods unique to each endoscopic performance task (A, DR, H, IM, N).

"""
Unified management of differences in the performance of A, DR, H, IM and N.
It includes: legal category list, scoring range of model prompt words, regular matching logic, and splicing rules of few shot examples.
"""
TASK_CONFIG = {
    "A": {
        "classes": ["A0", "A1", "A2", "NO"],
        "score_range": "C-0, C-1, C-2, C-3, O-1, O-2, O-3, NO",
        "regex": r"\b(C-[0-3]|O-[1-3]|NO)\b",
        "format_example": lambda ex: (
            f"1.视角：{ex['view']}\n"
            f"2.位置：{ex['location']}\n"
            f"3.是否存在萎缩：{ex['atrophy']}\n"
            f"4.是否观察到萎缩边界：{ex['boundary']}\n"
            f"5.图片评分：{ex['score'].replace('图片评分：', '').replace('。', '').replace('（无表现）', '').strip().upper()}"
        ),
        "reasoning": "结合视角、位置、黏膜颜色变化、厚度、血管显露及萎缩边界情况，与木村竹本分型标准一致。"
    },
    "DR": {
        "classes": ["DR0", "DR1", "DR2", "NO"],
        "score_range": "DR-0, DR-1, DR-2, NO",
        "regex": r"\b(DR-0|DR-1|DR-2|NO)\b",
        "format_example": lambda ex: (
            f"1.位置：{ex['location']}\n"
            f"2.描述：{ex['atrophy']}\n"
            f"3.图片评分：{ex['score'].replace('图片评分：', '').replace('。', '').strip().upper()}"
        ),
        "reasoning": "结合静脉的规则排列（RAC）与DR评分标准。"
    },
    "H": {
        "classes": ["H0", "H1", "NO"],
        "score_range": "H-0, H-1, NO",
        "regex": r"\b(H-0|H-1|NO)\b",
        "format_example": lambda ex: (
            f"1.视角：{ex['view']}\n"
            f"2.位置：{ex['location']}\n"
            f"3.描述：{ex['atrophy']}\n"
            f"4.图片评分：{ex['score'].replace('图片评分：', '').replace('。', '').strip().upper()}"
        ),
        "reasoning": "结合皱襞宽度与H评分标准。"
    },
    "IM": {
        "classes": ["IM0", "IM1", "IM2", "NO"],
        "score_range": "IM-0, IM-1, IM-2, NO",
        "regex": r"\b(IM-0|IM-1|IM-2|NO)\b",
        "format_example": lambda ex: (
            f"1.图像类型：{ex['view']}\n"
            f"2.位置：{ex['location']}\n"
            f"3.有无肠上皮化生：{ex['atrophy']}\n"
            f"4.分级：{ex['classify']}\n"
            f"5.图片评分：{ex['score'].replace('图片评分：', '').replace('。', '').strip().upper()}"
        ),
        "reasoning": "结合图片是否存在白色隆起病变与IM评分标准。"
    },
    "N": {
        "classes": ["N0", "N1", "no"],
        "score_range": "N-0, N-1, NO",
        "regex": r"\b(N-0|N-1|NO)\b",
        "format_example": lambda ex: (
            f"1.位置：{ex['location']}\n"
            f"2.描述：{ex['atrophy']}\n"
            f"3.图片评分：{ex['score'].replace('图片评分：', '').replace('。', '').strip().upper()}"
        ),
        "reasoning": "结合胃窦黏膜是否隆起与N评分标准。"
    }
}
