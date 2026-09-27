# -*- coding: utf-8 -*-
"""LangGraph 多 Agent 辩论工作流。

流程：
  1. generate   —— 三套方案（没有则现场并行生成，已生成则直接进入辩论）
  2. advocate   —— 每套方案一个「辩护 Agent」立论（并行）
  3. rebuttal   —— 每套方案一个「攻辩 Agent」攻击另外两套并回应质疑（并行，可多轮）
  4. moderator  —— 「主持人 Agent」听完全场辩论，给出排名与一套合并方案

所有 Agent 都只在信息内讨论，不直接改动项目数据；合并方案由前端「采用」后写入。
"""

import concurrent.futures

from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from novel_generator.unit_generation import (
    _extract_json,
    _make_adapter,
    _normalize_proposal,
    generate_proposals,
    invoke_with_cleaning,
)


class DebateState(TypedDict, total=False):
    reference_text: str
    guidance: str
    target_words: int
    proposals: list
    model: dict
    rebuttal_rounds: int
    cases: dict
    rebuttals: dict
    verdict: dict
    merged: dict


def _proposal_text(proposal) -> str:
    proposal = proposal if isinstance(proposal, dict) else {}
    world = proposal.get("world") if isinstance(proposal.get("world"), dict) else {}
    world_lines = [
        f"类型：{world.get('genre', '') or '—'}",
        f"前提：{world.get('premise', '') or '—'}",
        f"时代与地点：{world.get('era_place', '') or '—'}",
        f"规则：{world.get('rules', '') or '—'}",
        f"基调：{world.get('tone', '') or '—'}",
    ]
    characters = proposal.get("characters") if isinstance(proposal.get("characters"), list) else []
    char_lines = [f"- {item.get('name', '（未命名）')}：{item.get('notes', '')}" for item in characters if isinstance(item, dict)]
    directory = proposal.get("directory") if isinstance(proposal.get("directory"), list) else []
    step_lines = [f"{index}. {item.get('who', '')} → {item.get('what', '')}" for index, item in enumerate(directory, start=1) if isinstance(item, dict)]
    return "\n".join([
        f"方案 {proposal.get('key', '?')}《{proposal.get('title', '未命名')}》",
        f"卖点：{proposal.get('thesis', '') or '—'}",
        f"单主线：{proposal.get('mainline', '') or '—'}",
        "世界观：" + "；".join(world_lines),
        "人物：" + ("；".join(char_lines) if char_lines else "（无）"),
        "大事件目录：" + ("；".join(step_lines) if step_lines else "（无）"),
    ])


def _truncate(text: str, limit: int = 1600) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def build_advocate_prompt(*, proposal, guidance: str, target_words: int) -> str:
    return f"""你是小说策划评审 Agent，此刻担任「方案 {proposal.get('key')}」的辩护律师。
你的唯一任务：论证你负责的这套方案是三套里最值得采用的一套，写得扎实、具体，不空洞客套。

【作者意图】{guidance or '（未说明）'}
【全书体量】约 {target_words or '未指定'} 字

【你负责的方案全文】
{_truncate(_proposal_text(proposal))}

请输出：
1. 核心优势：这套方案相对另外两套最突出的 3 个优势，逐个结合正文细节说（悬念/情感线/结构/世界观可执行性皆可）。
2. 自检风险：这套方案最明显的 2 个弱点，以及怎么改可以化解。
3. 一句结论：为什么作者应该选它。

直接输出观点，不要重复方案原文。
""".strip()


def build_rebuttal_prompt(*, proposal, others, cases: dict, round_index: int) -> str:
    def other_block(item):
        key = item.get("key", "?")
        case = _truncate(str((cases or {}).get(key, "") or "（未立论）"))
        return f"<方案 {key}>\n正文：{_truncate(_proposal_text(item), 2000)}\n其立论：{case}\n</方案 {key}>"
    return f"""你是小说策划评审 Agent，此刻担任「方案 {proposal.get('key')}」的攻辩手。辩题是：三套方案哪一套最该被作者采用。
这是第 {round_index} 轮交锋。你的任务：
1. 犀利地指出另外两套方案的硬伤：主线逻辑漏洞、与作者意图的偏离、世界观不可自洽、目录推进乏力，每套至少 2 条，必须引用其正文细节，禁止空洞攻击。
2. 回应别人可能对你所持方案的质疑：挑出本方案最容易被攻击的 1～2 点，主动给出回击。
3. 陈述经过这一轮之后，你所持方案仍然最强的理由。

【你负责的方案】
{_truncate(_proposal_text(proposal), 2000)}

【对方方案与立论】
{chr(10).join(other_block(item) for item in others)}

直接输出论点，不要寒暄。
""".strip()


def build_moderator_prompt(*, proposals, cases: dict, rebuttals: dict, reference_text: str, guidance: str, target_words: int) -> str:
    rounds_text = []
    for proposal in proposals:
        key = proposal.get("key", "?")
        case = _truncate(str((cases or {}).get(key, "") or ""))
        rebuttal_list = (rebuttals or {}).get(key) or []
        rebuttal_text = "\n\n".join(f"第 {index} 轮：{_truncate(round_text)}" for index, round_text in enumerate(rebuttal_list, start=1))
        rounds_text.append(
            f"<方案 {key}>\n正文：{_truncate(_proposal_text(proposal), 2200)}\n立论：{case}\n攻辩记录：{rebuttal_text or '（无）'}\n</方案 {key}>"
        )
    return f"""你是本场辩论的主持人 Agent。三套方案已各自立论、互相攻辩完毕，现在轮到你做总结裁决。

【作者意图】{guidance or '（未说明）'}
【全书体量】约 {target_words or '未指定'} 字
【参考文本节选】{_truncate(reference_text or '（未上传）', 1200)}

【全部辩论记录】
{chr(10).join(rounds_text)}

请输出两个部分：
1. ranking：三套方案按推荐程度排序，从第 1 名到第 3 名，每套给出 score（满分 10）、strengths（优点）、weaknesses（弱点）、comment（一句话理由）。
2. merged：以排在前面的方案为基础，吸收辩论中被论证有效的优点、规避被指出来的硬伤，融合出一套最终方案。字段结构如下，且必须满足：
   - mainline 必须写清楚全部细节，绝对禁止写概述（主角/起点/目标/最大阻力/至少两个关键转折/情感走向/结局落点）；
   - directory 给出 8～14 个事件，每个事件只写「谁 + 做什么」，覆盖开场到结局。

只返回 JSON（不要任何其他文字）：
{{"decision":"A","ranking":[{{"key":"A","score":9,"strengths":"","weaknesses":"","comment":""}}],"merged":{{"key":"M","title":"","thesis":"","mainline":"","world":{{"genre":"","premise":"","era_place":"","rules":"","tone":""}},"characters":[{{"name":"","gender":"男","age":"24","notes":""}}],"directory":[{{"seq":1,"who":"","what":""}}]}}}}

decision 的取值规则：如果某套方案评分明显最高且无需大改，填它的 key（如 "A"）；如果融合后的 merged 明显优于任何一套，填 "M"。二者都给出，decision 只填一个值。
""".strip()


def _fan_out(run_one, items, workers):
    """并行执行 run_one(item)，每个任务返回 (key, value)；value 为空时不收录。"""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(int(workers or 3), 8))) as pool:
        futures = {pool.submit(run_one, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            try:
                pair = future.result()
            except Exception:
                continue
            if pair and pair[0] and pair[1]:
                results[pair[0]] = pair[1]
    return results


def _call_json(model, prompt, label):
    result = invoke_with_cleaning(_make_adapter(model), prompt, label=label, invoke_json=True)
    return _extract_json(result)


def _progress_hook(config):
    hook = (config or {}).get("configurable", {}).get("progress") if isinstance(config, dict) else None
    return hook if callable(hook) else None


def generate_node(state: DebateState, config=None) -> dict:
    progress = _progress_hook(config)
    if progress:
        progress("generate", "并行生成三套方案", "正在并行调用 3 次模型，各自生成一套全书方案…", 0, 3)
    proposals = state.get("proposals") or []
    if not proposals:
        result = generate_proposals(
            reference_text=state.get("reference_text", ""),
            guidance=state.get("guidance", ""),
            target_words=int(state.get("target_words") or 0),
            model=state.get("model") or {},
        )
        proposals = result.get("proposals", []) or []
    if progress:
        progress("generate", "并行生成三套方案", f"三套方案已生成（共 {len(proposals)} 套，每套含主线/世界观/人物/大事件目录）", len(proposals), 3)
    return {"proposals": proposals}


def advocate_node(state: DebateState, config=None) -> dict:
    progress = _progress_hook(config)
    proposals = [item for item in (state.get("proposals") or []) if isinstance(item, dict)]
    guidance = state.get("guidance", "")
    target_words = int(state.get("target_words") or 0)
    model = state.get("model") or {}

    if progress:
        progress("advocate", "三个 Agent 立论", "每个方案一个辩护 Agent，正在陈述核心优势与风险自检…", 0, len(proposals))

    done_count = [0]

    def run_one(proposal):
        key = str(proposal.get("key") or proposal.get("title") or "?")
        prompt = build_advocate_prompt(proposal=proposal, guidance=guidance, target_words=target_words)
        try:
            result = invoke_with_cleaning(_make_adapter(model), prompt, label=f"三方辩论：方案 {key} 立论", invoke_json=False)
            text = str(result or "").strip() or None
        except Exception:
            text = None
        done_count[0] += 1
        if progress:
            progress("advocate", "三个 Agent 立论", f"方案 {key} 已立论（{done_count[0]}/{len(proposals)}）", done_count[0], len(proposals))
        return key, text

    return {"cases": _fan_out(run_one, proposals, workers=3)}


def rebuttal_node(state: DebateState, config=None) -> dict:
    progress = _progress_hook(config)
    proposals = [item for item in (state.get("proposals") or []) if isinstance(item, dict)]
    cases = state.get("cases") or {}
    rounds = max(1, min(3, int(state.get("rebuttal_rounds") or 1)))
    model = state.get("model") or {}
    rebuttals = {str(item.get("key")): [] for item in proposals}

    for round_index in range(1, rounds + 1):
        if progress:
            progress("rebuttal", "三个 Agent 攻辩", f"第 {round_index} 轮攻辩：互相质疑并回应质疑…", 0, len(proposals))
        done_count = [0]

        def run_one(proposal):
            key = str(proposal.get("key") or "?")
            others = [item for item in proposals if item is not proposal]
            prompt = build_rebuttal_prompt(proposal=proposal, others=others, cases=cases, round_index=round_index)
            try:
                result = invoke_with_cleaning(_make_adapter(model), prompt, label=f"三方辩论：方案 {key} 攻辩 第{round_index}轮", invoke_json=False)
                text = str(result or "").strip() or None
            except Exception:
                text = None
            done_count[0] += 1
            if progress:
                progress("rebuttal", "三个 Agent 攻辩", f"第 {round_index} 轮：方案 {key} 已发言（{done_count[0]}/{len(proposals)}）", done_count[0], len(proposals))
            return key, text

        round_results = _fan_out(run_one, proposals, workers=3)
        for item in proposals:
            key = str(item.get("key"))
            text = round_results.get(key)
            if text:
                rebuttals[key].append(text)

    return {"rebuttals": rebuttals}


def moderator_node(state: DebateState, config=None) -> dict:
    progress = _progress_hook(config)
    if progress:
        progress("moderator", "主持人裁决", "正在评分排名，必要时融合出一套新方案…", 0, 1)
    proposals = [item for item in (state.get("proposals") or []) if isinstance(item, dict)]
    model = state.get("model") or {}
    parsed = _call_json(
        model,
        build_moderator_prompt(
            proposals=proposals,
            cases=state.get("cases") or {},
            rebuttals=state.get("rebuttals") or {},
            reference_text=state.get("reference_text", ""),
            guidance=state.get("guidance", ""),
            target_words=int(state.get("target_words") or 0),
        ),
        label="三方辩论：主持人裁决与合并方案",
    )
    if not isinstance(parsed, dict):
        parsed = {}
    ranking = parsed.get("ranking") if isinstance(parsed.get("ranking"), list) else []
    rank_keys = [str(item.get("key", "") or "") for item in ranking if isinstance(item, dict)]
    decision = str(parsed.get("decision", "") or "").strip()
    if decision not in rank_keys + ["M"]:
        decision = rank_keys[0] if rank_keys else "M"
    verdict = {
        "ranking": ranking,
        "summary": str(parsed.get("summary", "") or "").strip(),
        "decision": decision,
    }
    merged = _normalize_proposal("M", parsed.get("merged"))
    if not merged and proposals:
        merged = dict(proposals[0])
        merged["key"] = "M"
        merged["title"] = f"{merged.get('title', '')}（合并）".strip("（合并）")
    if progress:
        progress("moderator", "主持人裁决", f"裁决完成：推荐方案 {verdict.get('decision', '?')}", 1, 1)
    return {"verdict": verdict, "merged": merged}


def build_debate_graph():
    builder = StateGraph(DebateState)
    builder.add_node("generate", generate_node)
    builder.add_node("advocate", advocate_node)
    builder.add_node("rebuttal", rebuttal_node)
    builder.add_node("moderator", moderator_node)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "advocate")
    builder.add_edge("advocate", "rebuttal")
    builder.add_edge("rebuttal", "moderator")
    builder.add_edge("moderator", END)
    return builder.compile()


def run_debate(*, reference_text: str = "", guidance: str = "", target_words: int = 0, proposals=None, model: dict, rebuttal_rounds: int = 1, progress=None) -> dict:
    """执行辩论工作流，返回立论、攻辩记录、主持人裁决与合并方案。"""
    graph = build_debate_graph()
    initial = {
        "reference_text": reference_text,
        "guidance": guidance,
        "target_words": int(target_words or 0),
        "proposals": [dict(item) for item in (proposals or []) if isinstance(item, dict)],
        "model": model,
        "rebuttal_rounds": max(1, min(3, int(rebuttal_rounds or 1))),
        "cases": {},
        "rebuttals": {},
        "verdict": {},
        "merged": {},
    }
    invoke_config = {"configurable": {"progress": progress}} if callable(progress) else None
    final = graph.invoke(initial, config=invoke_config) if invoke_config else graph.invoke(initial)
    return {
        "proposals": final.get("proposals") or [],
        "cases": final.get("cases") or {},
        "rebuttals": final.get("rebuttals") or {},
        "verdict": final.get("verdict") or {},
        "merged": final.get("merged"),
    }
