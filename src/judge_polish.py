"""
Secondary pass: judge model (qwen3p7-plus via Fireworks — benchmark runner-up,
fastest, and a different family from the glm-5p2 caption writer to avoid
self-preference bias) used to (a) pick the best caption per style out of the
Best-of-N candidates, (b) polish humor-style captions, and (c) self-critique
every caption before submission. This whole module is optional-by-design: if
Fireworks is unavailable or errors, callers must fall back to the base
caption rather than fail the clip.
"""
import json
import re

import requests

import config
from prompts import (
    JUDGE_POLISH_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    PICK_BEST_SYSTEM_PROMPT,
    build_judge_polish_prompt,
    build_judge_prompt,
    build_pick_best_prompt,
)


class JudgeAssistant:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, model: str | None = None):
        self.api_key = api_key or config.FIREWORKS_API_KEY
        self.base_url = base_url or config.FIREWORKS_BASE_URL
        self.model = model or config.FIREWORKS_JUDGE_MODEL

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        # temperature defaults to 0.0: pick-best and critique are JUDGING
        # tasks — the same caption must always get the same verdict, or the
        # critique loop polishes captions on coin flips. polish() overrides
        # with 0.7 because rewriting is a creative task.
        # max_tokens 450: a polished 50-word formal caption plus the judge's
        # JSON both fit; the old 250 could truncate a polish rewrite.
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 450,
                "temperature": temperature,
                "reasoning_effort": "none",
            },
            timeout=config.PER_REQUEST_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    def pick_best(self, style: str, scene_hint: str, candidates: list) -> str:
        """Given N candidate captions for one style, returns the judge's
        pick. Falls back to the first candidate on any error — never lets
        Best-of-N selection break the pipeline."""
        candidates = [c for c in candidates if c]
        if not candidates:
            return ""
        if len(candidates) == 1 or not self.available:
            return candidates[0]
        try:
            raw = self._chat(PICK_BEST_SYSTEM_PROMPT, build_pick_best_prompt(style, scene_hint, candidates))
            idx = int(_extract_json(raw).get("best", 1)) - 1
            choice = candidates[idx] if 0 <= idx < len(candidates) else candidates[0]
            print(f"[pick-best] {style} via {self.model}: chose candidate {idx + 1}/{len(candidates)}")
            return choice
        except Exception as e:
            print(f"[pick-best] {style} via {self.model}: FAILED ({e}) — keeping first candidate")
            return candidates[0]

    def polish(self, style: str, scene_hint: str, draft_caption: str) -> str:
        if not self.available:
            return draft_caption
        try:
            result = self._chat(
                JUDGE_POLISH_SYSTEM_PROMPT,
                build_judge_polish_prompt(style, scene_hint, draft_caption),
                temperature=0.7,
            )
            print(f"[judge-polish] {style} via {self.model}: OK "
                  f"({len(draft_caption)} -> {len(result)} chars)")
            return result
        except Exception as e:
            print(f"[judge-polish] {style} via {self.model}: FAILED ({e}) — keeping draft caption")
            return draft_caption  # never let polish failures break the pipeline

    def judge(self, style: str, scene_hint: str, caption: str):
        """Returns (score: float, feedback: str). Defaults to a passing
        score if the judge call itself fails, so we don't retry forever
        on infra errors."""
        if not self.available:
            return 10.0, ""
        try:
            raw = self._chat(JUDGE_SYSTEM_PROMPT, build_judge_prompt(style, scene_hint, caption))
            data = _extract_json(raw)
            score, feedback = float(data.get("score", 10)), str(data.get("feedback", ""))
            print(f"[judge-critique] {style} via {self.model}: OK (score={score})")
            return score, feedback
        except Exception as e:
            print(f"[judge-critique] {style} via {self.model}: FAILED ({e}) — defaulting to pass")
            return 10.0, ""


def _extract_json(text: str) -> dict:
    """Same robust extraction as fireworks_vision_client: tolerates trailing
    junk after the JSON object and JSON embedded in prose."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        # Parsed but not an object — fall through to object extraction below.
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"no JSON object found in: {text[:200]}")
