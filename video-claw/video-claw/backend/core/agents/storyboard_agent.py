# -*- coding: utf-8 -*-
"""
阶段3: 分镜智能体
基于剧本JSON，逐场景拆分为带时长标签的分镜（shots），按幕分组输出。
支持 Segment -> Shots 嵌套结构。
"""

import os
import re
import json
import asyncio
import logging
from datetime import datetime
from typing import Any, Optional, Dict, List, Tuple

from .base_agent import AgentInterface

logger = logging.getLogger(__name__)

def _get_shot_prompt(lang: str = "zh") -> str:
    from prompts.loader import load_prompt_with_fallback
    return load_prompt_with_fallback("storyboard", "shot", lang, "zh")

class StoryboardAgent(AgentInterface):
    def __init__(self):
        super().__init__(name="Storyboard")

    MIN_SHOT_DURATION = 3
    MIN_SEGMENT_DURATION = 5
    MAX_SEGMENT_DURATION = 15
    OPENING_SHOT_TYPES = {"中景", "全景"}

    @staticmethod
    def _read_script_json(sid: str) -> dict:
        session_path = os.path.join("code/data/sessions", f"{sid}.json")
        if not os.path.exists(session_path):
            return {}
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("artifacts", {}).get("script_generation", {})
        except Exception:
            return {}

    @staticmethod
    def _extract_json_array(text: str) -> Optional[List[dict]]:
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            result = json.loads(text)
            if isinstance(result, list): return result
        except json.JSONDecodeError: pass
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                if isinstance(result, list): return result
            except json.JSONDecodeError: pass
        return None

    @staticmethod
    def _clean_script_line(line: str) -> str:
        line = line.strip()
        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^<action>\s*", "", line, flags=re.I)
        line = re.sub(r"\s*</action>$", "", line, flags=re.I)
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    @staticmethod
    def _strip_markup(text: str) -> str:
        text = re.sub(r"^[#>\s]+", "", text.strip())
        text = text.strip("*_` \t")
        return text.strip()

    @staticmethod
    def _character_names(characters: List[dict]) -> List[str]:
        names = []
        for item in characters:
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
        return sorted(set(names), key=len, reverse=True)

    @staticmethod
    def _setting_names(settings: List[dict]) -> List[str]:
        names = []
        for item in settings:
            name = str(item.get("name", "")).strip()
            if name:
                names.append(name)
        return sorted(set(names), key=len, reverse=True)

    @classmethod
    def _match_characters(cls, text: str, character_names: List[str]) -> List[str]:
        """Only use the existing character list as keywords; never invent names."""
        return [name for name in character_names if name and name in text]

    @staticmethod
    def _setting_matches_text(setting_name: str, text: str) -> bool:
        if setting_name in text:
            return True
        normalized = re.sub(r"[（）()【】\[\]\s]", "", setting_name)
        text_normalized = re.sub(r"\s", "", text)
        if normalized and normalized in text_normalized:
            return True
        base_name = re.sub(r"[（(].*?[）)]", "", setting_name).strip()
        return bool(base_name and base_name in text)

    @classmethod
    def _resolve_location(
        cls,
        text: str,
        current_location: str,
        setting_names: List[str],
    ) -> str:
        for name in setting_names:
            if cls._setting_matches_text(name, text):
                return name
        if current_location:
            return current_location
        return setting_names[0] if setting_names else ""

    @classmethod
    def _parse_scene_header(cls, line: str, setting_names: List[str]) -> Optional[str]:
        clean = cls._strip_markup(line)
        if not clean:
            return None

        # Examples:
        # **第1集-第1场 日 内 高二三班教室**
        # **1-1 夜 内 锐创科技办公室**
        header_patterns = [
            r"^第\d+集\s*[-—]\s*第\d+场\s+(?:日|夜|晨|傍晚|深夜)\s+(?:内|外)\s+(.+)$",
            r"^\d+\s*[-—_]\s*\d+\s+(?:日|夜|晨|傍晚|深夜)\s+(?:内|外)\s+(.+)$",
        ]
        for pattern in header_patterns:
            match = re.match(pattern, clean, flags=re.I)
            if not match:
                continue
            candidate = match.group(1).strip()
            for name in setting_names:
                if cls._setting_matches_text(name, candidate):
                    return name
            return candidate[:40] or clean[:40]
        return None

    @staticmethod
    def _is_metadata_line(line: str) -> bool:
        return bool(re.match(r"^(?:剧本名称|时长|风格|类型|标题)[:：]", line))

    @staticmethod
    def _is_end_marker(line: str) -> bool:
        return bool(re.match(r"^[（(]?(?:第.+集\s*)?完[）)]?$|^\(?THE END\)?$", line.strip(), flags=re.I))

    @staticmethod
    def _dialogue_parts(line: str) -> Optional[Tuple[str, str, str]]:
        """Return speaker, tone/action, dialogue when a line looks like dialogue."""
        match = re.match(r"^([^:：]{1,24}?)(?:[（(]([^）)]{0,40})[）)])?[:：]\s*(.+)$", line)
        if not match:
            return None
        speaker = match.group(1).strip()
        tone = (match.group(2) or "").strip()
        dialogue = match.group(3).strip()
        if not dialogue:
            return None
        # Avoid treating section labels as dialogue.
        if speaker in {"人物", "场景", "画面", "镜头", "地点", "时间"}:
            return None
        return speaker, tone, dialogue

    @classmethod
    def _estimate_duration(cls, text: str, is_dialogue: bool) -> int:
        """Heuristic duration in seconds; every shot is at least 3s."""
        visible_text = re.sub(r"[“”\"'，。！？、,.!?；;：:\s]", "", text)
        if is_dialogue:
            duration = 3 + len(visible_text) // 18
        else:
            duration = 3 + len(visible_text) // 34
        return max(cls.MIN_SHOT_DURATION, min(duration, cls.MAX_SEGMENT_DURATION))

    @staticmethod
    def _infer_shot_type(text: str, is_dialogue: bool) -> str:
        if re.search(r"城市|办公室|窗外|全景|拉远|场景|夜景|两台电脑|屏幕并排", text):
            return "全景"
        if re.search(r"特写|屏幕|手机|键盘|终端|报表|手指|代码|PASS|眼睛|嘴角|表情", text):
            return "近景"
        if is_dialogue:
            return "中景"
        return "中景"

    @classmethod
    def _normalize_first_shot_type(cls, shots: List[dict]) -> None:
        if not shots:
            return
        if shots[0].get("shot_type") not in cls.OPENING_SHOT_TYPES:
            shots[0]["shot_type"] = "中景"
            content = shots[0].get("content", "")
            shots[0]["content"] = re.sub(r"^(近景|过肩近景|特写|大特写)", "中景", content, count=1)

    @classmethod
    def _ensure_segment_duration(cls, shots: List[dict]) -> int:
        total = sum(int(shot.get("duration") or cls.MIN_SHOT_DURATION) for shot in shots)
        if shots and total < cls.MIN_SEGMENT_DURATION:
            shots[-1]["duration"] += cls.MIN_SEGMENT_DURATION - total
            total = cls.MIN_SEGMENT_DURATION
        return min(total, cls.MAX_SEGMENT_DURATION)

    @classmethod
    def _make_shot_content(
        cls,
        *,
        shot_type: str,
        line: str,
        characters: List[str],
        is_dialogue: bool,
        speaker: str = "",
        tone: str = "",
    ) -> str:
        if is_dialogue:
            tone_text = tone or "自然、贴合当下情绪"
            return f"{shot_type}，镜头对准{speaker or '角色'}，呈现动作与表情变化。{speaker}说：“{line}”。音色：{tone_text}。"
        subject = "、".join(characters) if characters else "场景主体"
        return f"{shot_type}，镜头呈现{subject}。{line}"

    @classmethod
    def _segment_from_shots(
        cls,
        ep_n: int,
        segment_number: int,
        location: str,
        shots: List[dict],
        characters: List[str],
    ) -> dict:
        for idx, shot in enumerate(shots, 1):
            shot["shot_number"] = idx
            shot["duration"] = min(
                cls.MAX_SEGMENT_DURATION,
                max(cls.MIN_SHOT_DURATION, int(shot.get("duration") or cls.MIN_SHOT_DURATION)),
            )
        cls._normalize_first_shot_type(shots)
        total_duration = cls._ensure_segment_duration(shots)
        return {
            "segment_id": f"seg_{ep_n:02d}_{segment_number:02d}",
            "segment_number": segment_number,
            "total_duration": total_duration,
            "location": location,
            "characters": characters,
            "shots": shots,
            "episode_number": ep_n,
        }

    @classmethod
    def _build_segments_by_regex(
        cls,
        ep_n: int,
        script_text: str,
        characters: List[dict],
        settings: List[dict],
    ) -> List[dict]:
        """Deterministically parse script text into model-ready segments.

        This path intentionally mirrors the LLM prompt output:
        segment_number, total_duration, location, characters, shots[].
        If it cannot extract useful shots, the caller falls back to LLM.
        """
        character_names = cls._character_names(characters)
        setting_names = cls._setting_names(settings)
        raw_lines = [cls._strip_markup(line) for line in script_text.replace("\r\n", "\n").split("\n")]

        atomic_shots: List[dict] = []
        current_location = setting_names[0] if setting_names else ""
        scene_characters: List[str] = []
        scene_key = 0
        found_scene_header = False

        for raw_line in raw_lines:
            line = cls._clean_script_line(raw_line)
            if not line or cls._is_metadata_line(line) or cls._is_end_marker(line):
                continue
            if re.match(r"^第?\d+集$", line):
                continue

            header_location = cls._parse_scene_header(line, setting_names)
            if header_location:
                found_scene_header = True
                scene_key += 1
                current_location = header_location
                matched = cls._match_characters(line, character_names)
                if matched:
                    scene_characters = matched
                continue

            if re.match(r"^人物[:：]", line):
                scene_characters = cls._match_characters(line, character_names)
                continue

            matched_chars = cls._match_characters(line, character_names)
            shot_chars = matched_chars or scene_characters
            dialogue = cls._dialogue_parts(line)
            is_dialogue = dialogue is not None
            speaker = ""
            tone = ""
            shot_text = line
            if dialogue:
                speaker, tone, shot_text = dialogue
                speaker_matches = cls._match_characters(speaker, character_names)
                if speaker_matches:
                    shot_chars = list(dict.fromkeys(speaker_matches + shot_chars))
                elif speaker in {"旁白", "独白", "画外音"}:
                    shot_chars = shot_chars or scene_characters

            shot_type = cls._infer_shot_type(line, is_dialogue)
            duration = cls._estimate_duration(shot_text, is_dialogue)
            location = cls._resolve_location(line, current_location, setting_names)

            atomic_shots.append({
                "scene_key": scene_key,
                "location": location,
                "characters": shot_chars,
                "shot": {
                    "shot_number": 0,
                    "shot_type": shot_type,
                    "duration": duration,
                    "content": cls._make_shot_content(
                        shot_type=shot_type,
                        line=shot_text,
                        characters=shot_chars,
                        is_dialogue=is_dialogue,
                        speaker=speaker,
                        tone=tone,
                    ),
                },
            })

        if not found_scene_header or not atomic_shots:
            return []

        segments: List[dict] = []
        current_items: List[dict] = []
        current_scene_key: Optional[int] = None
        current_location = ""
        current_chars: List[str] = []

        def flush_current():
            nonlocal current_items, current_scene_key, current_location, current_chars
            if not current_items:
                return
            shots = [item["shot"] for item in current_items]
            chars: List[str] = []
            for item in current_items:
                for name in item["characters"]:
                    if name not in chars:
                        chars.append(name)
            if not chars:
                chars = current_chars
            segments.append(cls._segment_from_shots(
                ep_n,
                len(segments) + 1,
                current_location,
                shots,
                chars,
            ))
            current_items = []
            current_scene_key = None
            current_location = ""
            current_chars = []

        for item in atomic_shots:
            item_duration = int(item["shot"]["duration"])
            current_duration = sum(int(existing["shot"]["duration"]) for existing in current_items)
            scene_changed = current_items and (
                item["scene_key"] != current_scene_key or item["location"] != current_location
            )
            would_overflow = current_items and current_duration + item_duration > cls.MAX_SEGMENT_DURATION

            # A scene switch always starts a new video-model call. Otherwise greedily
            # pack shots until the next one would exceed the 15s upper bound.
            if scene_changed or would_overflow:
                flush_current()

            current_items.append(item)
            current_scene_key = item["scene_key"]
            current_location = item["location"]
            for name in item["characters"]:
                if name not in current_chars:
                    current_chars.append(name)

        flush_current()
        return segments

    @classmethod
    def _normalize_llm_segments(cls, ep_n: int, extracted: List[dict]) -> List[dict]:
        valid_segments = []
        for seg in extracted:
            if not isinstance(seg, dict):
                continue

            shots = seg.get("shots", [])
            pending_shots = []
            for s in shots:
                if not isinstance(s, dict):
                    continue
                dur = s.get("duration", cls.MIN_SHOT_DURATION)
                try:
                    dur = int(dur)
                except (TypeError, ValueError):
                    dur = cls.MIN_SHOT_DURATION
                pending_shots.append({
                    "shot_number": 0,
                    "shot_type": s.get("shot_type", "中景"),
                    "duration": max(cls.MIN_SHOT_DURATION, dur),
                    "content": s.get("content", "")
                })

            if not pending_shots:
                continue

            chunk: List[dict] = []

            def flush_chunk():
                nonlocal chunk
                if not chunk:
                    return
                valid_segments.append(cls._segment_from_shots(
                    ep_n,
                    len(valid_segments) + 1,
                    seg.get("location", ""),
                    chunk,
                    seg.get("characters", []),
                ))
                chunk = []

            for shot in pending_shots:
                chunk_duration = sum(int(item.get("duration") or cls.MIN_SHOT_DURATION) for item in chunk)
                if chunk and chunk_duration + int(shot["duration"]) > cls.MAX_SEGMENT_DURATION:
                    flush_chunk()
                chunk.append(shot)
            flush_chunk()
        return valid_segments

    @staticmethod
    def _validate_episodes(episodes: List[dict]) -> List[dict]:
        """验证嵌套的 Episode -> Segment -> Shots 结构"""
        valid_episodes = []
        for ep in episodes:
            if not isinstance(ep, dict): continue
            
            segments = ep.get("segments", [])
            valid_segments = []
            for idx, seg in enumerate(segments, 1):
                if not isinstance(seg, dict): continue
                
                shots = seg.get("shots", [])
                valid_shots = []
                calc_total_duration = 0
                
                for s in shots:
                    if not isinstance(s, dict): continue
                    dur = s.get("duration", 5)
                    calc_total_duration += dur
                    valid_shots.append({
                        "shot_number": s.get("shot_number", len(valid_shots) + 1),
                        "shot_type": s.get("shot_type", "中景"),
                        "duration": dur,
                        "content": s.get("content", "")
                    })
                
                valid_segments.append({
                    "segment_id": seg.get("segment_id", f"seg_{str(idx).zfill(8)}"),
                    "segment_number": seg.get("segment_number", len(valid_segments) + 1),
                    "total_duration": seg.get("total_duration", calc_total_duration),
                    "location": seg.get("location", ""),
                    "characters": seg.get("characters", []),
                    "shots": valid_shots
                })
            
            valid_episodes.append({
                "episode_number": ep.get("episode_number", len(valid_episodes) + 1),
                "episode_title": ep.get("episode_title", ""),
                "segments": valid_segments
            })
        return valid_episodes

    async def process(self, input_data: Any, intervention: Optional[Dict] = None) -> Dict:
        from models.llm_client import LLM
        input_data = self._merge_session_params(input_data)
        sid = input_data.get("session_id")
        if not sid: raise Exception("Missing session_id")
             
        session_file = os.path.join("code/data/sessions", f"{sid}.json")
        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)
            
        llm_model = input_data.get("llm_model") or session_data.get("llm_model")
        if not llm_model:
            raise ValueError("Missing required model configuration: llm_model")
        style = input_data.get("style") or session_data.get("style") or "anime"
        
        # 处理人工干预/修改
        if intervention and "modified_storyboard" in intervention:
            modified_episodes = intervention["modified_storyboard"]
            if isinstance(modified_episodes, str): modified_episodes = json.loads(modified_episodes)
            session_data.setdefault("artifacts", {})["storyboard"] = {
                "session_id": sid,
                "episodes": modified_episodes,
                "user_modified": True,
                "updated_at": datetime.now().isoformat()
            }
            # 更新顶层状态
            session_data["updated_at"] = datetime.now().timestamp()
            with open(session_file, "w", encoding="utf-8") as f: json.dump(session_data, f, indent=2, ensure_ascii=False)
            return {"payload": {"session_id": sid, "episodes": modified_episodes}, "stage_completed": True}
        
        script_data = self._read_script_json(sid)
        if not script_data: raise Exception("未找到剧本数据")
        
        episodes = script_data.get("episodes", [])
        if not episodes:
            raise Exception("剧本数据中不包含有效集数列表(episodes)")

        # 检查是否有已存在的分镜数据，识别需要生成的集数
        existing_storyboard = session_data.get("artifacts", {}).get("storyboard", {})
        existing_story_eps = existing_storyboard.get("episodes", [])
        
        # 建立已生成的 segments 索引
        ready_eps = {e["episode_number"] for e in existing_story_eps if e.get("segments")}
        
        # 确定需要处理的集数：如果该集还没有 segments，则需要生成
        episodes_to_proc = [ep for ep in episodes if ep.get("episode_number") not in ready_eps]
        
        if not episodes_to_proc:
            logger.info("[Storyboard] All episodes already have storyboard segments. Skipping generation.")
            return {"payload": {"session_id": sid, "episodes": existing_story_eps}, "stage_completed": True}

        chars = script_data.get("characters", [])
        sets = script_data.get("settings", [])
        is_zh = any("\u4e00" <= c <= "\u9fff" for c in script_data.get("title", ""))
        shot_prompt_tpl = _get_shot_prompt("zh" if is_zh else "en")
        
        self._report_progress("分镜", f"开始生成 {len(episodes_to_proc)} 集缺失的分镜...", 5)
        
        async def proc_ep(ep):
            # ... (保持原本处理逻辑不变)
            ep_n = ep.get("episode_number", 1)
            # ... (以下为原本 proc_ep 逻辑)
            ep_t = ep.get("act_title", f"第{ep_n}集")
            ep_c = ep.get("content", "")

            regex_segments = self._build_segments_by_regex(ep_n, ep_c, chars, sets)
            if regex_segments:
                logger.info("[Storyboard] Episode %s parsed by regex into %d segments.", ep_n, len(regex_segments))
                self._report_progress("分镜", f"集数 {ep_n} 已通过规则解析生成分镜", 35)
                return {
                    "episode_number": ep_n,
                    "episode_title": ep_t,
                    "segments": regex_segments
                }

            logger.warning("[Storyboard] Episode %s regex parsing produced no segments; falling back to LLM.", ep_n)
            
            # 清洗剧本内容：按行拆分，用于提示说明“一行一分镜”
            lines = [line.strip() for line in ep_c.split('\n') if line.strip()]
            script_text_with_lines = "\n".join([f"L{idx+1}: {line}" for idx, line in enumerate(lines)])

            prompt = shot_prompt_tpl.format(
                act_number=ep_n, 
                act_title=ep_t, 
                script_text=script_text_with_lines, 
                asset_characters=json.dumps(chars, ensure_ascii=False), 
                asset_settings=json.dumps(sets, ensure_ascii=False),
                style=style
            )
            
            llm = LLM()
            loop = asyncio.get_running_loop()
            
            # 增加重试机制 (Max 3 retries)
            max_retries = 3
            extracted = None
            raw = ""
            
            for attempt in range(max_retries):
                try:
                    self._report_progress("分镜", f"生成集数 {ep_n} (第 {attempt + 1} 次尝试)...", 5 + attempt * 2)
                    raw = await loop.run_in_executor(None, self._cancellable_query, llm, prompt, [], llm_model, False, sid, False)
                    extracted = self._extract_json_array(raw)
                    if extracted:
                        break
                    logger.warning(f"Episode {ep_n} Attempt {attempt + 1}: Failed to extract JSON array. Retrying...")
                except Exception as e:
                    logger.error(f"Episode {ep_n} Attempt {attempt + 1}: LLM query error: {str(e)}")
                    if attempt == max_retries - 1: raise e
            
            if not extracted:
                logger.error(f"LLM output failed to parse as JSON for Episode {ep_n} after {max_retries} attempts. Raw: {raw}")
                # 抛出异常以触发 orchestrator 的错误状态处理逻辑
                raise Exception(f"第 {ep_n} 集分镜生成失败：模型输出无法解析")
                
            valid_segments = self._normalize_llm_segments(ep_n, extracted)
            if not valid_segments:
                raise Exception(f"第 {ep_n} 集分镜生成失败：无法从模型输出构造有效片段")

            return {
                "episode_number": ep_n,
                "episode_title": ep_t,
                "segments": valid_segments
            }

        # 核心：支持流式保存增量产物，让前端能看到实时进度
        updated_ep_map = {e["episode_number"]: e for e in existing_story_eps}
        
        # 立即先保存一次，确保已有的 episodes 在进入 running 状态后依然可见
        session_data.setdefault("artifacts", {})["storyboard"] = {
            "session_id": sid, 
            "episodes": sorted(updated_ep_map.values(), key=lambda x: x["episode_number"]),
            "created_at": datetime.now().isoformat()
        }
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        # 报告一次进度，带上 asset_complete 强制前端从磁盘刷新一次初步数据
        self._report_progress("分镜设计", "准备生成分镜...", 10, {"asset_complete": True})

        results_queue = [proc_ep(ep) for ep in episodes_to_proc]

        for coro in asyncio.as_completed(results_queue):
            res = await coro
            updated_ep_map[res["episode_number"]] = res
            
            # 每完成一集分镜，立即持久化并触发增量同步
            temp_eps = sorted(updated_ep_map.values(), key=lambda x: x["episode_number"])
            session_data.setdefault("artifacts", {})["storyboard"] = {
                "session_id": sid, 
                "episodes": temp_eps, 
                "created_at": datetime.now().isoformat()
            }
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            # 报告带有 asset_complete 的进度，强制编排器刷新前端数据
            self._report_progress("分镜设计", f"集数 {res['episode_number']} 分镜已生成", 50, {"asset_complete": True})

        final_all_episodes = sorted(updated_ep_map.values(), key=lambda x: x["episode_number"])
        
        # 核心：将结果持久化到 artifacts.storyboard
        session_data.setdefault("artifacts", {})["storyboard"] = {
            "session_id": sid, 
            "episodes": final_all_episodes, 
            "created_at": datetime.now().isoformat()
        }
        
        session_data["updated_at"] = datetime.now().timestamp()

        with open(session_file, "w", encoding="utf-8") as f: 
            json.dump(session_data, f, indent=2, ensure_ascii=False)
            
        self._report_progress("分镜", "完成", 100)
        return {"payload": {"session_id": sid, "episodes": final_all_episodes}, "stage_completed": True}
