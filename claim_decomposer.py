"""主张拆分模块：将长新闻文本拆分为多条独立事实主张。

支持：
- LLM 智能拆分
- 规则验证（确保不遗漏、不新增虚假主张）
- 去重和合并相似主张
"""

from __future__ import annotations

import re
from typing import Optional

from src.llm.client import LLMClient, LLMError
from src.models import Claim


# 拆分 Prompt
DECOMPOSE_SYSTEM_PROMPT = """你是一个事实核查专家。请将输入的新闻文本拆分为多条独立、可核查的事实主张。

要求：
1. 每条主张必须是独立的、可单独核查的事实陈述
2. 不要遗漏原文中的任何关键事实
3. 不要新增原文没有的信息
4. 每条主张应简洁清晰，避免模糊描述
5. 区分不同的事实：人物身份、时间、地点、事件、数据等
6. 如果一个句子包含多个事实，需要拆分

输出严格的 JSON 格式：
{
  "claims": [
    {
      "text": "主张文本（完整的事实陈述）",
      "claim_type": "事实/观点/预测/统计数据",
      "priority": 1,
      "reason": "为什么这条重要"
    }
  ]
}"""

DECOMPOSE_USER_PROMPT_TEMPLATE = """请将以下新闻文本拆分为多条独立的事实主张：

{text}

注意：
- 如果文本较短（<50字），可能只有1-2条主张
- 如果文本较长或包含多个事实，请尽可能详细地拆分
- 按重要性排序，最重要的主张优先级为1"""


def decompose_claims_llm(text: str, max_claims: int = 8) -> list[dict]:
    """使用 LLM 将文本拆分为多条主张。

    Args:
        text: 输入文本
        max_claims: 最大主张数量（默认8条，支持更长文本）

    Returns:
        主张列表，每个包含 text, claim_type, priority, reason
    """
    if not text or not text.strip():
        return []

    # 短文本直接作为单条主张
    if len(text.strip()) <= 30:
        return [{
            "text": text.strip(),
            "claim_type": "事实",
            "priority": 1,
            "reason": "短文本，单条主张",
        }]

    try:
        client = LLMClient(timeout=30)  # 增加超时时间

        # 处理长文本：超过2000字时分块处理
        if len(text) > 2000:
            return _decompose_long_text(text, max_claims, client)

        user_prompt = DECOMPOSE_USER_PROMPT_TEMPLATE.format(text=text[:2000])

        raw = client.chat(
            system_prompt=DECOMPOSE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=0.1,
        )

        # 解析 JSON
        import json
        cleaned = _strip_code_fence(raw)
        data = json.loads(cleaned)
        claims = data.get("claims", [])

        # 验证和过滤
        valid_claims = []
        for c in claims[:max_claims]:
            if isinstance(c, dict) and c.get("text", "").strip():
                valid_claims.append({
                    "text": c["text"].strip(),
                    "claim_type": c.get("claim_type", "事实"),
                    "priority": min(c.get("priority", 3), 5),
                    "reason": c.get("reason", ""),
                })

        # 确保至少有一条主张
        if not valid_claims:
            valid_claims = [{
                "text": text.strip(),
                "claim_type": "事实",
                "priority": 1,
                "reason": "LLM 拆分失败，使用原文",
            }]

        return valid_claims

    except (LLMError, Exception) as e:
        print(f"[Decompose] LLM 拆分失败，使用规则降级: {e}")
        return decompose_claims_rule_based(text, max_claims)


def _decompose_long_text(text: str, max_claims: int, client: LLMClient) -> list[dict]:
    """处理长文本（超过2000字）的拆分。

    策略：将文本分成多个块，分别处理后合并。
    """
    # 按句子分割
    sentences = re.split(r'([。！？!?\n])', text)
    chunks = []
    current_chunk = ""

    for sent in sentences:
        if not sent:
            continue
        if len(current_chunk) + len(sent) <= 1800:
            current_chunk += sent
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent

    if current_chunk:
        chunks.append(current_chunk.strip())

    # 分别处理每个块
    all_claims = []
    for chunk in chunks:
        if not chunk:
            continue
        try:
            user_prompt = DECOMPOSE_USER_PROMPT_TEMPLATE.format(text=chunk)
            raw = client.chat(
                system_prompt=DECOMPOSE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.1,
            )

            import json
            cleaned = _strip_code_fence(raw)
            data = json.loads(cleaned)
            chunk_claims = data.get("claims", [])

            for c in chunk_claims[:3]:  # 每个块最多3条
                if isinstance(c, dict) and c.get("text", "").strip():
                    all_claims.append({
                        "text": c["text"].strip(),
                        "claim_type": c.get("claim_type", "事实"),
                        "priority": c.get("priority", 3),
                        "reason": c.get("reason", ""),
                    })
        except Exception as e:
            print(f"[Decompose] 块处理失败: {e}")
            # 降级：直接把这个块作为一条主张
            all_claims.append({
                "text": chunk[:100],
                "claim_type": "事实",
                "priority": 3,
                "reason": "块处理失败，降级为整体",
            })

    # 去重和限制数量
    merged = _remove_duplicates(all_claims)
    return merged[:max_claims]


def _remove_duplicates(claims: list[dict]) -> list[dict]:
    """去除重复的主张。"""
    if len(claims) <= 1:
        return claims

    unique = []
    for c in claims:
        is_dup = False
        for u in unique:
            # 检查文本相似度
            sim = _text_similarity(c["text"], u["text"])
            if sim > 0.6:
                is_dup = True
                break
        if not is_dup:
            unique.append(c)

    return unique


def _strip_code_fence(text: str) -> str:
    """去除代码块围栏。"""
    text = text.strip()
    if text.startswith("```"):
        # 移除开头的 ```json 或 ```
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        # 移除结尾的 ```
        text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def decompose_claims_rule_based(text: str, max_claims: int = 5) -> list[dict]:
    """基于规则的主张拆分（降级方案）。

    策略：
    1. 按句号、感叹号、问号分段
    2. 识别包含时间、地点、人物的子句
    3. 去重和合并
    """
    if not text or not text.strip():
        return []

    # 分割句子
    sentences = re.split(r'[。！？!?\n]+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

    # 如果只有一个句子或短文本
    if len(sentences) <= 1:
        return [{
            "text": text.strip(),
            "claim_type": "事实",
            "priority": 1,
            "reason": "单句主张",
        }]

    claims = []
    for i, sentence in enumerate(sentences[:max_claims]):
        # 判断主张类型
        claim_type = _classify_claim_type(sentence)
        # 提取优先级
        priority = _estimate_priority(sentence, i)

        claims.append({
            "text": sentence,
            "claim_type": claim_type,
            "priority": priority,
            "reason": f"第{i+1}句，基于规则拆分",
        })

    return claims


def _classify_claim_type(text: str) -> str:
    """规则判断主张类型。"""
    if any(kw in text for kw in ["预测", "将", "会", "计划", "预计"]):
        return "预测"
    if any(kw in text for kw in ["认为", "表示", "称", "说", "指出"]):
        return "观点"
    if any(kw in text for kw in ["%", "万", "亿", "数量", "统计"]):
        return "统计数据"
    return "事实"


def _estimate_priority(text: str, index: int) -> int:
    """估算优先级（1-5）。"""
    score = 3  # 默认中等

    # 包含时间、地点、人物的更重要
    if re.search(r'\d{4}年|\d{1,2}月|\d{1,2}日|今天|昨天|现在', text):
        score -= 1
    if any(kw in text for kw in ["官方", "正式", "宣布", "突发"]):
        score -= 1
    if re.search(r'\d+%|\d+万|\d+亿', text):
        score -= 1

    # 位置加权
    if index == 0:
        score -= 1

    return max(1, min(5, score))


def merge_overlapping_claims(claims: list[dict]) -> list[dict]:
    """合并重叠或相似的主张。

    使用简单的文本相似度检测。
    """
    if len(claims) <= 1:
        return claims

    merged = []
    for claim in claims:
        # 检查是否与已有主张相似
        is_duplicate = False
        for existing in merged:
            similarity = _text_similarity(claim["text"], existing["text"])
            if similarity > 0.7:
                is_duplicate = True
                break

        if not is_duplicate:
            merged.append(claim)

    return merged


def _text_similarity(text1: str, text2: str) -> float:
    """计算简单的 Jaccard 相似度。"""
    # 使用字符 n-gram
    def get_ngrams(text: str, n: int = 2) -> set[str]:
        text = text.lower()
        return {text[i:i+n] for i in range(len(text)-n+1)}

    ngrams1 = get_ngrams(text1)
    ngrams2 = get_ngrams(text2)

    if not ngrams1 or not ngrams2:
        return 0.0

    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2

    return len(intersection) / len(union) if union else 0.0


def claims_to_claim_objects(claims_data: list[dict]) -> list[Claim]:
    """将拆分结果转换为 Claim 对象列表。"""
    claims = []
    for i, data in enumerate(claims_data):
        claim = Claim(
            claim_id=f"claim_{i+1}",
            text=data["text"],
            claim_type=data.get("claim_type", "事实"),
            entities=[],
            verification_priority=data.get("priority", 3),
        )
        claims.append(claim)
    return claims
