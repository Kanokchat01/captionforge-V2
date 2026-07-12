"""
Prompt construction for the Track 2 Video Captioning Agent.

Two-stage pipeline, two Stage-1 paths (no audio on either):
  Stage 1 (primary):  whole clip via video_url + minimax-m3 ->
                      build_native_video_scene_analysis_prompt ->
                      structured 10-section Scene Report (sees real motion,
                      timelines, camera movement)
  Stage 1 (fallback): sampled still frames + build_frame_scene_analysis_prompt
                      -> same 10-section Scene Report
  Stage 2: Scene Report (text only) + style rules -> 4 captions JSON

Style keys MUST match the official spec exactly (underscore, not hyphen):
formal, sarcastic, humorous_tech, humorous_non_tech
"""
import hashlib
import re

# The OFFICIAL style definitions, verbatim from the Participant Guide's
# style table. The automated judge is almost certainly prompted with these
# exact one-liners, so every internal prompt (writer, pick-best, critique)
# leads with them rather than with our own invented constraints.
STYLE_DESCRIPTIONS = {
    "formal": "Professional, objective, factual tone",
    "sarcastic": "Dry, ironic, lightly mocking",
    "humorous_tech": "Funny, with technology or programming references",
    "humorous_non_tech": "Funny, everyday humour with no technical jargon",
}

# Numeric word-count ranges per style — MUST mirror the ranges written in
# STYLE_RULES below. Used for programmatic compliance checks (candidate
# filtering + polish revert guard) in main.py / web_demo. These are OUR
# soft targets, not contest rules — the official rubric never counts words,
# so the ranges are deliberately wide; they only exist to catch runaway
# rewrites, not to straitjacket the writer.
STYLE_WORD_RANGES = {
    "formal": (30, 50),
    "sarcastic": (15, 32),
    "humorous_tech": (18, 35),
    "humorous_non_tech": (18, 35),
}
DEFAULT_WORD_RANGE = (15, 35)


def stable_variety_index(key: str) -> int:
    """Kept for interface compatibility (web_demo passes it through). The
    per-clip opening/emoji rotation it used to select was removed 2026-07-12:
    captions are judged per clip in isolation, so cross-clip repetition was
    never visible to the judge, and none of the 0.92 leaderboard teams use
    emojis or forced openings at all. Python's built-in hash() is salted per
    process, so use a stable digest."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 4


# humorous_non_tech is DEFINED by the absence of tech ("Funny, everyday humour
# with no technical jargon"), so a single tech word in it directly costs the
# style-match half of that caption's score. The writer mostly obeys, but a
# 12-clip run still produced "a high-stakes game of human Tetris" — a video
# game reference inside the no-tech style. Best-of-N makes this cheap to fix
# in code: candidates carrying any of these words are dropped before the judge
# ever sees them (main.py), with a fallback to the full pool if every
# candidate trips the filter. Only unambiguous computing/gaming terms are
# listed — words like "screen", "system", or "bug" have innocent everyday
# senses and would cause false rejections.
TECH_JARGON_RE = re.compile(
    r"\b("
    r"wi-?fi|internet|online|offline|download(?:ing|ed)?|upload(?:ing|ed)?|"
    r"software|hardware|computer|laptop|keyboard|desktop|smartphone|"
    r"algorithm|coding|programming|program|debug(?:ging)?|compile[rd]?|"
    r"server|database|api|cache|latency|bandwidth|firmware|router|"
    r"buffering|glitch|render(?:ing|ed)?|pixel|byte|cpu|gpu|ram|usb|"
    r"respawn|spawn(?:s|ed|ing)?|npc|hitbox|speed-?run|pathfinding|"
    r"tetris|minecraft|app|gadget|robot(?:ic)?s?"
    r")\b",
    re.IGNORECASE,
)


def has_tech_jargon(text: str) -> bool:
    """True if a caption contains a term that would break humorous_non_tech's
    'no technical jargon' definition."""
    return bool(TECH_JARGON_RE.search(text))


# Emoji and pictographs (all styles ban them as of 2026-07-12 — none of the
# 0.92 leaderboard teams use emojis, the official style definitions never
# mention them, and the writer kept violating the old palette rules anyway).
EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # emoji, symbols, pictographs, supplemental
    "☀-➿"          # misc symbols + dingbats (☕ ⚠ ✅ ...)
    "⬀-⯿"          # arrows/stars (⭐ ...)
    "️"                 # variation selector
    "‼⁉™ℹ"
    "]+"
)

# Any script outside basic Latin — captions must be English only (contest
# rule). Deterministic, no LLM needed: one Thai/CJK/Cyrillic/Arabic/etc.
# character is an instant violation.
NON_ENGLISH_RE = re.compile(
    "["
    "Ͱ-Ͽ"  # Greek
    "Ѐ-ӿ"  # Cyrillic
    "֐-׿"  # Hebrew
    "؀-ۿ"  # Arabic
    "ऀ-෿"  # Devanagari..Sinhala
    "฀-๿"  # Thai
    "ᄀ-ᇿ"  # Hangul Jamo
    "぀-ヿ"  # Hiragana/Katakana
    "㐀-鿿"  # CJK
    "가-힯"  # Hangul syllables
    "]"
)

# humorous_tech is DEFINED by tech references, so a caption with NO tech term
# at all fails the style by definition. This is a broad REQUIRE-list (the
# mirror image of TECH_JARGON_RE's narrow ban-list): a false positive merely
# admits a candidate for the judge to rank, so generous matching is the safe
# direction.
TECH_TERMS_RE = re.compile(
    r"\b("
    r"wi-?fi|internet|online|offline|download\w*|upload\w*|stream(?:s|ed|ing)?|buffer\w*|"
    r"software|hardware|firmware|computer|laptop|desktop|keyboard|monitor|smartphone|phone|"
    r"apps?|update[sd]?|install\w*|reboot\w*|restart\w*|shut\s?down|power cycle|"
    r"algorithm\w*|code|coding|coded|program\w*|debug\w*|compil\w*|deploy\w*|dev|develop\w*|"
    r"server\w*|database|data|cloud|api|cache[ds]?|caching|latency|bandwidth|router|network\w*|"
    r"glitch\w*|lag(?:s|gy|ging|ged)?|render\w*|pixel\w*|bytes?|megabyte|gigabyte|cpu|gpu|ram|usb|"
    r"a\.?i\.?|bots?|robot\w*|automat\w*|machine[- ]learning|neural|"
    r"load(?:s|ed|ing)? screen|loading|errors?|bug(?:s|gy)?|crash\w*|40[34]|ctrl|alt[- ]tab|"
    r"spreadsheets?|emails?|browsers?|passwords?|login|logged (?:in|out|on|off)|log (?:in|out|on|off)|"
    r"sync\w*|backups?|cursor|scroll\w*|refresh\w*|ping(?:s|ed|ing)?|firewall|beta|patch(?:es|ed|ing)?|"
    r"screensaver|notifications?|airplane mode|low[- ]power mode|battery|charg(?:e|er|ing)|"
    r"version \d|v\d\.\d|systems?|process(?:es|or|ing)?|memory|storage|hard drive|ssd|"
    r"terminal|command line|git|merge conflict|pull request|stack ?trace|null|undefined|"
    r"infinite loop|loops?|queues?|threads?|kernel|framerate|fps|resolution"
    r")\b",
    re.IGNORECASE,
)

# Interjections that break sarcastic's "dry, ironic" register.
SARCASTIC_BANNED_RE = re.compile(
    r"\b(lo+l|ha(?:ha)+|omg|rofl|lmao)\b|no cap\b|fr fr\b|literally dying",
    re.IGNORECASE,
)


def style_violations(style: str, text: str) -> list:
    """Mechanical, deterministic style checks a caption must pass before it
    is worth the judge's time. Returns a list of short actionable problem
    strings (empty = compliant) — main.py uses them both to filter Best-of-N
    candidates and as polish feedback for a final repair attempt."""
    v = []
    if EMOJI_RE.search(text):
        v.append("remove every emoji")
    if NON_ENGLISH_RE.search(text):
        v.append("rewrite entirely in English")
    if style == "formal" and "!" in text:
        v.append("remove the exclamation mark")
    if style == "sarcastic" and SARCASTIC_BANNED_RE.search(text):
        v.append("remove the laughing/hype interjection; keep it dry and deadpan")
    if style == "humorous_tech" and not TECH_TERMS_RE.search(text):
        v.append("add one clear technology or programming term")
    if style == "humorous_non_tech" and has_tech_jargon(text):
        v.append("remove all technical jargon")
    return v


# Meta-labels the writer has been caught pasting into the caption itself when
# it mistook a prompt instruction for text to copy (real examples from a
# 12-clip run: "Opening with: Watching the tiny cars scurry below...",
# "Opening phrase: Let's be real,\nLet's be real, my fingers are moving...").
# The prompt wording that caused it is fixed, but a caption is the one artifact
# the judge reads — cheap to scrub as a last line of defence.
_LEAKED_LABEL_RE = re.compile(
    r"^\s*(opening(?:\s+(?:with|phrase|line))?|caption|answer|output|style)\s*[:\-–]\s*",
    re.IGNORECASE,
)


def sanitize_caption(text: str) -> str:
    """Strip emojis and prompt-instruction leakage, collapse newlines, drop
    stray wrapping quotes, and de-duplicate an opener the model repeated after
    a leaked label. Returns a single clean line."""
    cleaned = EMOJI_RE.sub("", str(text))  # belt-and-braces: no style carries emoji anymore
    cleaned = " ".join(cleaned.split())  # collapses newlines and runs of spaces
    for _ in range(2):  # a label can survive one strip, e.g. "Caption: Opening: ..."
        stripped = _LEAKED_LABEL_RE.sub("", cleaned)
        if stripped == cleaned:
            break
        cleaned = stripped
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in "\"'":
        cleaned = cleaned[1:-1].strip()
    # "Let's be real, Let's be real, my fingers..." -> keep one copy of the head
    head = re.match(r"^(.{4,40}?[,:])\s*\1\s*", cleaned)
    if head:
        cleaned = f"{head.group(1)} {cleaned[head.end():]}"
    return cleaned.strip()


def word_count(text: str) -> int:
    return len(text.split())


def in_word_range(style: str, text: str) -> bool:
    """True if the caption's word count sits inside the style's rule range."""
    lo, hi = STYLE_WORD_RANGES.get(style, DEFAULT_WORD_RANGE)
    return lo <= word_count(text) <= hi


# Full per-style generation rules used in Stage 2. Each entry LEADS with the
# official Participant Guide definition verbatim (what the real judge scores
# against); everything after it is craft guidance, kept deliberately looser
# than before — the automated judge scores accuracy + style match only, it
# never counts words or checks openings, so rigid structural rules only cost
# creativity without buying points.
STYLE_RULES = {
    "formal": (
        'FORMAL — official definition: "Professional, objective, factual '
        'tone." Voice: a wire-service news reporter — precise, neutral, '
        'zero opinion, zero flourish. Write 2-3 sentences, 30-50 words '
        'total, embedding at least THREE distinct concrete visual facts '
        'from the report (pick from: subject appearance, setting type, '
        'specific actions in order, camera perspective, lighting and '
        'colors, how many people/vehicles/animals are visible). No slang, '
        'no emoji, no exclamation marks, no contractions (write "does '
        'not", never "doesn\'t"), no jokes. Do not start with "The scene '
        'shows" or "The video captures," and do not end with a generic '
        'summary sentence. Describe the setting generically (e.g. "an '
        'urban street," "a modern office") — never name a specific '
        'real-world city, country, or landmark. You may name a playback '
        'effect (time-lapse, slow motion) ONLY if the analysis report '
        'itself states it; never name capture equipment or settings '
        '(drone, lens type, exposure).'
    ),
    "sarcastic": (
        'SARCASTIC — official definition: "Dry, ironic, lightly mocking." '
        'Voice: an exhausted observer who has watched this exact thing a '
        'thousand times — eye-roll energy delivered completely flat. '
        '15-32 words. Deadpan understatement beats hype: short, flat '
        'sentences in subject-verb-object order. Mock at least TWO '
        'specific, visible things from THIS clip — the irony comes from '
        'the matter-of-fact delivery of an absurd or mundane observation, '
        'never from hype or exclamation. No emoji, no exclamation marks, '
        'no laughing interjections (haha, lol), no loud Gen-Z hype slang '
        '("main character energy," "aura points," "cooked," "no cap") — '
        'a light touch of contemporary phrasing is fine if it stays '
        'understated. A dry, lightly mocking punchline ending lands well.'
    ),
    "humorous_tech": (
        'HUMOROUS_TECH — official definition: "Funny, with technology or '
        'programming references." Voice: a burnt-out software engineer '
        'cracking a joke about the scene. 18-35 words. Build ONE clear '
        'tech or programming joke on something actually happening in the '
        'video — name the real subject in plain words and weave at least '
        'TWO visible details from the report into the joke (the actual '
        'animal, vehicle, or person, what it looks like, what it is '
        'doing). Compare the scene TO tech; do NOT rewrite the scene AS '
        'software — the literal subject and action must stay readable '
        'through the joke, or the caption fails on accuracy ("the dog '
        'sprints like it heard the dinner-bell notification" works; "the '
        'canine process executes a pathing loop" does not). At least one '
        'unmistakable technology or programming term must appear. A '
        'generic tech pun that could be pasted onto any video is a FAIL. '
        'One core joke, punchy delivery, no emoji.'
    ),
    "humorous_non_tech": (
        'HUMOROUS_NON_TECH — official definition: "Funny, everyday humour '
        'with no technical jargon." Voice: your funny relative who has '
        'never owned a smartphone — warm, relatable, draws comparisons '
        'from cooking, weather, chores, pets, and family life. 18-35 '
        'words. Ground the joke in at least TWO visible details from the '
        'report and match the scenario to the video\'s domain (exhaustion '
        'for sports, hunger for food, social dread for interviews, '
        'weather-ruined plans...). The joke must be a real human moment, '
        'not a restatement of the scene. An opening like "When you..." or '
        '"That feeling when..." is allowed but optional — a plain '
        'first-person or observational line is just as good. Zero '
        'technical content of any kind: this persona does not know '
        'computing, internet, gaming, or engineering words exist, and '
        'never references video games (a line like "a game of human '
        'Tetris" already breaks this style). No emoji.'
    ),
}

def build_frame_scene_analysis_prompt(frame_timestamps, video_duration):
    """Stage 1 prompt for the frame-based path. The model receives N still
    frames as images, at the given timestamps, and NO audio at all. Produces
    a 10-section Scene Report; section 6 (AUDIO) is forced to "No audio
    present" instead of asking the model to guess at sound it never got, so
    build_caption_generation_prompt (Stage 2) needs no changes."""
    ts_list = ", ".join("{:.1f}s".format(t) for t in frame_timestamps)
    header = (
        "You are a professional video analyst. You are given "
        + str(len(frame_timestamps))
        + " still frames sampled from a "
        + "{:.0f}".format(video_duration)
        + "-second video, in chronological order, taken at approximately "
        "these timestamps: " + ts_list + ". You do NOT have the audio "
        "track — do not guess at or invent any sound, dialogue, or music."
    )
    return header + """

If the frames cannot be analyzed (corrupted, blank, no visual content), write only: "ANALYSIS FAILED: [brief reason]" and stop. Do not guess or invent content you cannot actually see.

Pay close attention to: visual details, actions implied by how the scene changes between frames, camera work, lighting, and mood. Treat gaps between frames as unknown — do not invent what happened between them.

Keep each section concise — 2-4 sentences, except Key Actions and Standout Details which may use short bullet points. Avoid padding with generic description.

Write your report in the following 10 sections:

--- SCENE REPORT ---

1. SUBJECT
Who or what is the main focus? Describe appearance in detail (species, color, size, clothing, expression, distinguishing features).
When describing eyes/gaze, report only what is directly visible. If the eyes are lowered, downcast, shadowed, or not clearly wide open, describe it as "looking down" or "gaze directed at [whatever is in front of them]" — do NOT assert the eyes are closed or that the subject is drowsy, sleeping, or dozing off. While a subject is actively doing a task (typing, reading, eating, working), lowered or hidden eyes almost always mean the gaze is aimed at that task, not that the eyes are shut. Only state the eyes are closed if a frame shows both eyelids unmistakably and fully shut with no task-directed gaze possible.

2. ENVIRONMENT
Where does this take place? Describe the setting, surfaces, objects, background elements, weather, and time of day. Describe the TYPE of place generically (e.g. "a multi-lane urban street," "an office with desks and computer monitors"). Do NOT name a specific real-world city, country, neighborhood, or landmark unless it is explicitly, legibly written on a sign or screen in the frames — guessing a location from general visual style (architecture, plant species, signage style) is exactly the kind of unconfirmed claim that must NOT appear here.

3. KEY ACTIONS (timeline)
List what changes across the sampled frames, chronologically, using the frame timestamps given above.
Format: [MM:SS] Action/description at that frame.
Example: [00:03] A kitten sits behind leafy branches, looking at the camera.
Describe only the state actually visible in each sampled frame. Do not narrate a smooth or gradual transition that you are inferring to connect the frames (e.g. "slowly closes her eyes," "gradually speeds up," "begins to tire," "starts to fall asleep") — if the sampled frames do not unambiguously show that progression, report each frame's state on its own and leave what happens between them as unknown.

4. CAMERA & FRAMING
Describe the camera angle (low, high, eye-level) and framing (close-up, wide shot, depth of field) as seen across the frames. Only describe movement (pan/tilt/tracking) if it can be confidently inferred from how framing changes between frames — otherwise say "static or unknown." Describe only what is visibly evident (e.g. "vehicle lights appear as streaks," "background is blurred"). Do NOT name the specific photographic technique or equipment that supposedly produced it (e.g. "long exposure," "drone shot," "time-lapse," "macro lens," "slow motion") — naming a capture technique is a guess about how the footage was made, not an observation of what is shown.

5. LIGHTING & COLOR
Describe the dominant light source, color palette, contrast, and any notable visual effects (lens flare, bokeh, golden hour glow, neon).

6. AUDIO
No audio track was provided for analysis. Write exactly: "No audio present." Do not guess at implied or expected sounds.

7. MOOD & ATMOSPHERE
What emotion do the frames evoke? (e.g., peaceful, chaotic, tense, heartwarming, eerie, comedic)

8. STANDOUT DETAILS
List 3-5 specific, quirky, or memorable details that make this video unique. These are the best ingredients for humor and captions.
Example: "The kitten's fur is backlit by sunlight, creating a golden halo effect."

9. HUMOR POTENTIAL
What is naturally funny, ironic, cute, dramatic, or absurd about this video? Think like a meme creator. Identify the 'comedy goldmine' moments.
Base this only on what is visually confirmed in sections 1-5 — not on assumed intent, thoughts, or emotions the subject cannot literally express. If a subject "looks annoyed," describe the visible expression, don't assert the subject IS annoyed.

10. RISKS (things NOT confirmed)
List anything a caption writer might assume or hallucinate that is NOT actually shown across the sampled frames, including anything that might have happened in the gaps between frames, any specific real-world location/city/landmark name that was NOT confirmed by legible on-screen text, any specific photographic technique or equipment that was NOT confirmed, any on-screen text/signage that appeared anywhere in the frames but was NOT 100% clearly legible (move it here instead of quoting it elsewhere, and describe it generically there too), and note that no audio was available.
Example: "No butterflies visible. No other animals. No human hands shown. No audio track analyzed. City/location not identifiable from the frames. Capture technique (e.g. exposure settings) not identifiable from the frames. Background building signage is present but too small/distant to read with certainty — do not quote it."

--- END REPORT ---

Important:
- Be specific, not generic. "Orange tabby kitten" not just "a cat."
- Describe what you actually SEE in these specific frames, not what you assume happens between them.
- Do not upgrade an ambiguous pose into a definite action or inner state — describe the literal, observable position instead. A lowered head or downward gaze is "looking down," NOT "eyes closing" or "falling asleep"; a crouched animal is "crouched low," NOT necessarily "about to pounce"; dark clouds are "dark clouds," NOT necessarily "an incoming storm"; a runner mid-step is "mid-stride," NOT necessarily "tiring." Only make the stronger claim if a frame shows it beyond any doubt.
- If unsure about something, say "possibly" or "appears to be." Anything marked this way should be treated as unconfirmed, not fact.
- On-screen text (signs, labels, screens): the bar for "clearly legible" is very high — every single letter must be sharp, large, high-contrast, and unmistakable, with zero doubt about any character. In wide shots, distant background signage, small building signs among many others, or anything even slightly blurred, angled, small, or partially occluded, you do NOT have enough information — describe it generically instead ("a building sign", "storefront signage", "illuminated signage in the background") and do NOT quote, paraphrase, or partially reconstruct the wording. If you are not 100% certain of every single word, treat the ENTIRE sign as illegible — never transcribe part of it and guess the rest, and never "clean up" a fuzzy read into a plausible-sounding phrase. A misread sign quoted in a caption directly costs accuracy points and is always worse than a generic description, even if the generic version sounds less impressive.
"""


def build_native_video_scene_analysis_prompt() -> str:
    """Stage 1 prompt for the native-video path: the model receives the WHOLE
    clip via video_url (minimax-m3), so unlike the frame path it can describe
    continuous motion, real timelines, and camera movement. Same 10-section
    Scene Report structure as the frame path, so Stage 2 needs no changes.

    Two hard lessons baked in:
    - No audio reaches the model over video_url (verified 2026-07-11), so
      section 6 stays forced to "No audio present."
    - Native-video OCR is confidently WRONG (read a real "KOREA ILLIES
      ENGINEERING" building sign as "KOREA MEDIA ENGINEERING"), so on-screen
      text transcription is banned absolutely — no legibility exception like
      the frame prompt had."""
    return """You are a professional video analyst. You are given a full video clip (typically 30 seconds to 2 minutes long). You do NOT have the audio track — analyze the visuals only; do not guess at or invent any sound, dialogue, or music.

If the video cannot be analyzed (corrupted, blank, no visual content), write only: "ANALYSIS FAILED: [brief reason]" and stop. Do not guess or invent content you cannot actually see.

Pay close attention to: visual details, actions as they unfold over time, camera movement, lighting, and mood. You watched the whole clip, so describe real motion and real transitions — but never upgrade an ambiguous pose or moment into a definite inner state.

Keep each section concise — 2-4 sentences, except Key Actions and Standout Details which may use short bullet points. Avoid padding with generic description.

Write your report in the following 10 sections:

--- SCENE REPORT ---

1. SUBJECT
Who or what is the main focus? Describe appearance in detail (species, color, size, clothing, expression, distinguishing features). State HOW MANY people, animals, or vehicles are visible — an exact count if countable, otherwise an estimate ("about a dozen cars," "a crowd of roughly twenty"). Name the dominant colors of the main subject.
Report gaze and eyes only as directly visible. While a subject is actively doing a task (typing, reading, eating), lowered or hidden eyes almost always mean the gaze is aimed at that task — only state the eyes are closed if the video shows it unmistakably.

2. ENVIRONMENT
Where does this take place? Describe the setting, surfaces, objects, background elements, weather, and time of day. Name the dominant colors and materials of the setting, and describe the SPATIAL LAYOUT: what sits in the foreground vs the background, and what is on the left vs the right of frame. Describe the TYPE of place generically (e.g. "a multi-lane urban street," "an office with desks and computer monitors"). Do NOT name a specific real-world city, country, neighborhood, or landmark — guessing a location from visual style (architecture, plants, signage style, language on signs) is exactly the kind of unconfirmed claim that must NOT appear here.

3. KEY ACTIONS (timeline)
Describe chronologically what actually happens across the clip, from beginning to middle to end, with approximate timestamps.
Format: [MM:SS] Action/description around that moment.
Example: [00:03] A kitten steps out from behind leafy branches and walks toward the camera.
You watched the motion happen, so continuous actions and transitions ARE fair to describe (e.g. "traffic gradually backs up," "she turns from one monitor to the other") — but describe observable behavior only, never assumed intent or inner states.

4. CAMERA & FRAMING
Describe the camera angle (low, high, eye-level), framing (close-up, wide shot, depth of field), and camera MOVEMENT as actually seen: static, panning, tilting, dollying/pushing in or out, tracking a subject, zooming. Getting camera movement right matters — apparent motion of foreground objects is often the camera moving, not the objects. You may name a playback-speed effect (time-lapse, slow motion) ONLY when the motion itself shows it unmistakably. Never guess at capture equipment (drone, crane, gimbal, lens type, exposure settings) — that is production speculation, not observation.

5. LIGHTING & COLOR
Describe the dominant light source, color palette, contrast, and any notable visual effects (lens flare, bokeh, golden hour glow, neon).

6. AUDIO
No audio track was provided for analysis. Write exactly: "No audio present." Do not guess at implied or expected sounds.

7. MOOD & ATMOSPHERE
What emotion does the clip evoke? (e.g., peaceful, chaotic, tense, heartwarming, eerie, comedic)

8. STANDOUT DETAILS
List 5-8 specific, quirky, or memorable details that make this video unique — including things that only motion reveals (a repeated gesture, a rhythm in the traffic, a sudden stop). These are the best ingredients for humor and captions; the more concrete and countable, the better.

9. HUMOR POTENTIAL
What is naturally funny, ironic, cute, dramatic, or absurd about this video? Think like a meme creator. Identify the 'comedy goldmine' moments, especially ones the motion itself creates. Base this only on what is visually confirmed in sections 1-5 — if a subject "looks annoyed," describe the visible expression, don't assert the subject IS annoyed.

10. RISKS (things NOT confirmed)
List anything a caption writer might assume or hallucinate that is NOT actually shown: real-world location/city/landmark names (never confirmable), capture equipment (never confirmable), the wording of ANY on-screen text (never transcribable — see rule below; note here that signs/text are present but must stay generic), and that no audio was available.
Example: "No other animals appear. Location not identifiable. Capture equipment not identifiable. A building sign and storefront text are visible but their wording must NOT be quoted. No audio track analyzed."

--- END REPORT ---

Important:
- Be specific, not generic. "Orange tabby kitten" not just "a cat."
- Describe what you actually SEE. Do not upgrade an ambiguous moment into a definite claim: a lowered head is "looking down," NOT "falling asleep"; a crouched animal is "crouched low," NOT necessarily "about to pounce."
- If unsure about something, say "possibly" or "appears to be." Anything marked this way is unconfirmed, not fact.
- ON-SCREEN TEXT (signs, labels, screens, clothing print): video compression and downscaling make text unreliable — you WILL misread it even when it looks perfectly clear to you. NEVER transcribe, quote, paraphrase, or partially reconstruct ANY on-screen text anywhere in this report. No exceptions, not even for large, sharp, prominent titles. Describe such text generically instead ("a building sign," "a storefront banner," "text on the monitor"), optionally with position/size/color, and record its presence in RISKS.
"""


def build_report_verification_prompt(report: str) -> str:
    """Stage 1.5 (native path only): the model re-watches the SAME clip
    alongside its own draft report and strips or fixes anything it cannot
    confirm on the second viewing. Deletion-only by design — the pass may
    remove hallucinations but can never introduce new ones, which is what
    makes it safe to run unsupervised."""
    return f"""You are a meticulous fact-checker re-watching a video clip. Below is a draft scene report written after a first viewing of this same clip. Verify every claim in it against what the video actually shows, then output a corrected version of the SAME report.

--- DRAFT REPORT ---
{report}
--- END DRAFT REPORT ---

Rules for the corrected report:
- Keep the exact same structure and the same 10 numbered section headers (1. SUBJECT through 10. RISKS), starting with "--- SCENE REPORT ---" and ending with "--- END REPORT ---".
- Verify every specific claim: counts of people/animals/vehicles, colors, actions and their order, timestamps, camera angle and camera movement, lighting.
- DELETE any claim you cannot confirm on this viewing, or soften it to "possibly ..." if only partially supported.
- FIX any claim that is wrong (wrong count, wrong color, wrong direction, wrong order, wrong movement).
- Move anything you deleted or doubted into section 10 RISKS so the caption writer knows not to use it.
- ADD NOTHING NEW. No new details, no new interpretations, no new humor angles. You may only delete, soften, correct, or keep.
- Keep section 6 AUDIO exactly as "No audio present."
- On-screen text stays absolutely banned: never transcribe, quote, or paraphrase any text visible in the video, no matter how clear it looks.

Output ONLY the corrected report, nothing before or after it."""


CAPTION_GENERATION_SHARED_RULES = """If the video analysis lacks sufficient detail for a style's word count, write a shorter, purely factual caption instead of inventing content to fill the length.

HOW YOU ARE SCORED: the judge compares each caption against the actual video on two axes — factual accuracy and style match. A caption that correctly names several specific visual details from THIS clip scores higher on accuracy than a short generic line that could sit under many different videos. The formal caption must embed at least THREE distinct concrete visual facts from the report; every other style at least TWO, woven naturally into the joke or observation rather than listed. Every fact must come from the report — if the report is thin, embed fewer facts rather than inventing any.

WHICH FACTS TO USE: the judge re-watches the video independently, and two viewings of the same clip can disagree about small things. Anchor every caption in facts nobody could possibly get wrong: what the main subject is and its overall color, what kind of place this is, the single most obvious action, the lighting or weather, the camera framing as stated in the report. NEVER let a caption's central claim rest on fragile details: the exact color of a small object, jewelry or accessory specifics, a subtle or momentary movement, a brief gesture or expression by one person in a group (a forehead wipe, a nod, a laugh at one moment), an event visible for only an instant, exact counts of distant background objects, or anything the RISKS section flags. When several people are visible, describe what the group or the main person clearly does across the WHOLE clip rather than pinning a momentary action on one individual. The CENTRAL claim, mock target, or comparison of every caption must be the clip's main subject and its main action or situation — never background clutter, props, or side details (cables on a desk, an object at the edge of frame). One small quirky detail may appear as garnish, plainly stated by the report, but if it vanished the caption must still stand. Never quote exact durations, timestamps, or second-counts in a caption.

VARIETY ACROSS THE SET: the four captions are read side by side. Each caption must build on DIFFERENT concrete details of the clip where possible, and no two captions may share the same sentence structure or the same opening words.

Rules:
- Write every caption in English only, regardless of any language seen or implied in the video.
- Write like a sharp human writer, NOT like AI or a textbook.
- Use strong, specific verbs that match what's actually happening in THIS video (examples only, do not default to these every time: chase, navigate, glow, weave — vary your verb choice based on the actual footage).
- No emoji or emoticons in any style.
- No inner double quotes. Use single quotes if needed.
- No questions, no hashtags, no call-to-action, no markdown.
- Before finalizing, count the words in each caption and confirm it fits the required range for that style.

Grounding: Every claim must come from the video analysis report above. Check the RISKS section of the report — do not include anything flagged there as uncertain or unconfirmed. Never name a real-world city, country, or landmark.

ON-SCREEN TEXT — ABSOLUTE RULE: never quote, transcribe, or paraphrase the wording of any sign, label, screen, or clothing text in a caption, even if the report mentions text exists. Automated OCR of compressed video misreads text confidently, and a caption quoting misread text is scored as a factual error, while a generic description ("a building sign," "text on the screen") is never penalized. Always describe text generically.

BANNED WORDS (never use — they sound like AI):
bustling, captivating, showcases, delves, vibrant, tapestry, multifaceted, realm

WRONG vs RIGHT FOR SARCASTIC (Dry and Deadpan, Not Hype Slang):
❌ "The kitten struts in dripping with main character energy, aura points maxed, we are so cooked" — loud hype slang, not dry, zero visual facts
✅ "The orange kitten squeezes through the same leafy gap twice, then stares at the camera like the garden owes it an explanation." — flat, deadpan, mocks two specific visible details
❌ "Traffic is absolutely giving main character syndrome right now, no cap" — hype-speak, not mockery
✅ "Six lanes of evening traffic crawl past the glass towers. Everyone still drives like they are late for something important." — dry observation built on concrete facts

WRONG vs RIGHT FOR HUMOROUS_NON_TECH (Specific and Relatable, Not a Scene Summary):
❌ "When the feline navigates the green garden foliage" — just re-describes the scene, no joke
✅ "That tiny cat slides between the flower pots the way I slide past relatives at family dinners — slow, careful, and silently judging everyone." — two visible details plus a real human moment
❌ "Me trying to see the beautiful landscape and mountains in the quiet nature" — generic, no relatable angle
✅ "The fog ate the entire mountain view about a minute after the climb ended, which is exactly the thanks hiking gives you." — concrete details, everyday humor
"""


def build_caption_generation_prompt(scene_report: str, styles: list, variety_index: int = 0) -> str:
    """Stage 2 prompt: text-only, works purely from the Stage 1 Scene Report
    (no video re-attached — cheaper and faster than a second video call).

    `variety_index` is kept for interface compatibility (main.py/web_demo
    still pass it) but no longer selects anything — the per-clip opening and
    emoji assignments were removed 2026-07-12."""
    style_blocks = "\n\n".join(STYLE_RULES.get(s, f"{s.upper()} (15-35 words): write in this requested style.") for s in styles)
    keys_example = ", ".join(f'"{s}": "..."' for s in styles)
    return f"""Using ONLY the following video analysis report, generate {len(styles)} caption(s).

--- VIDEO ANALYSIS REPORT ---
{scene_report}
--- END REPORT ---

{CAPTION_GENERATION_SHARED_RULES}

Styles required for this clip:

{style_blocks}

Output JSON only with exactly these keys: {{{keys_example}}}
"""


JUDGE_POLISH_SYSTEM_PROMPT = (
    "You rewrite video captions to land their requested style more sharply, "
    "without changing the underlying facts. You are given the style rules, "
    "the video's scene details, and a draft caption (sometimes with reviewer "
    "feedback appended — address it). For humor and sarcastic styles, "
    "sharper means funnier and more specific to this clip; for formal it "
    "means cleaner, more precise, more factual. Keep the rewrite grounded "
    "in the same concrete details — do not invent new facts about the "
    "video — and keep it inside the style's word range. Return only the "
    "rewritten caption text, no preamble."
)


def build_judge_polish_prompt(style: str, scene_hint: str, draft_caption: str) -> str:
    # Pass the full per-style rules (word count, persona, structure) — not
    # just the one-line description — so a polish pass can't drift the caption
    # out of the structural constraints Stage 2 wrote it under.
    rules = STYLE_RULES.get(style, STYLE_DESCRIPTIONS.get(style, ""))
    return (
        f"Style rules the rewritten caption MUST still satisfy:\n{rules}\n\n"
        f"Scene details: {scene_hint}\n"
        f"Draft caption: {draft_caption}\n\n"
        "Rewrite it now — same facts, sharper execution of the required style."
    )


PICK_BEST_SYSTEM_PROMPT = (
    "You are the judge for a video-captioning contest. You are given the "
    "video's scene details and several numbered candidate captions for ONE "
    "requested style. The contest scores exactly two things, equally "
    "weighted: (a) accuracy — the caption faithfully reflects the described "
    "scene with no invented facts (inventing on-screen text wording or a "
    "real-world location is an automatic disqualifier); among accurate "
    "candidates, prefer the one that correctly embeds the most specific, "
    "verifiable visual details from the scene — a caption generic enough "
    "to fit many different videos loses to one that could only describe "
    "THIS video. And (b) style match — it genuinely lands in the "
    "officially defined style rather than being generic or AI-sounding. "
    "For the sarcastic and humorous styles, a genuinely sharp, funny, "
    "specific caption beats a safe generic one — but never at the cost of "
    "accuracy. Word counts are soft guidance, not disqualifiers. Respond "
    'with ONLY a JSON object: {"best": <1-based candidate number>}.'
)


def build_pick_best_prompt(style: str, scene_hint: str, candidates: list) -> str:
    # Give the picker the full style rules (which lead with the official
    # definition); the system prompt tells it to treat structural details
    # as soft guidance, so tone + accuracy decide the pick.
    rules = STYLE_RULES.get(style, STYLE_DESCRIPTIONS.get(style, ""))
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    return (
        f"Requested style: {style}\nStyle rules: {rules}\n"
        f"Scene details: {scene_hint}\n"
        f"Candidate captions:\n{numbered}"
    )


JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge scoring a video caption on a 0-10 scale for the "
    "contest's two official axes combined into one score, equally weighted: "
    "(a) accuracy — does it faithfully reflect the described scene with no "
    "invented facts (quoted on-screen text or a named real-world location "
    "is an automatic fail); reward captions that correctly embed several "
    "specific, verifiable visual details from the scene, and mark down "
    "captions generic enough to sit under many different videos. And (b) "
    "style match — does it genuinely land in the officially defined style; "
    "for sarcastic/humorous styles that means it is actually funny or "
    "biting about THIS clip, not generic filler. Do not deduct points for "
    "word count or opening phrasing. Respond with ONLY a JSON object: "
    "{\"score\": <0-10 number>, \"feedback\": \"<one short actionable "
    "sentence for how to improve it if score < 8>\"}."
)


def build_judge_prompt(style: str, scene_hint: str, caption: str) -> str:
    rules = STYLE_RULES.get(style, STYLE_DESCRIPTIONS.get(style, ""))
    return (
        f"Requested style: {style}\nStyle rules: {rules}\n"
        f"Scene details: {scene_hint}\n"
        f"Caption to judge: {caption}"
    )


# =========================================================================
# v6 qwen_direct prompts — one multimodal call per style, no describe stage.
#
# GEOMETRY IS LOAD-BEARING — DO NOT RESTRUCTURE. The shape (a short
# imperative persona of ~2 sentences + one numbered rules block + the
# literal <caption_output> tags, sent under a strict-formatter system
# prompt at temperature ~0.7 with reasoning off) is board-verified at
# 0.92-0.93; a rewrite to long roleplay personas, a softer system prompt,
# and renamed tags collapsed a comparable pipeline to 0.74. Adjust wording
# only, never the shape, and re-verify on the board after any change.
# =========================================================================

QWEN_DIRECT_SYSTEM_PROMPT = (
    "You turn a persona brief and a set of video frames into exactly one "
    "caption. Reply with plain English text only, and place the finished "
    "caption inside literal <caption_output> and </caption_output> tags. "
    "Never show your reasoning, never chat, never use markdown of any kind."
)

QWEN_DIRECT_FORMAT_RULES = (
    "\n\n### RESPONSE FORMAT ###\n"
    "1. Put one finished caption between <caption_output> and </caption_output> "
    "— the tags plus the caption are your entire reply.\n"
    "2. Write in English about what the frames visibly show; never quote "
    "on-screen text and never name a real city, country, or landmark.\n"
    "3. No emoji, no markdown, no notes or explanations before or after the tags."
)

QWEN_DIRECT_PERSONAS = {
    "formal": (
        "Caption these frames the way a wire-agency photo desk would: neutral, "
        "precise, and strictly observational. One polished declarative sentence "
        "stating what the footage shows — no opinion, no flourish."
    ),
    "sarcastic": (
        "React with dry, unimpressed sarcasm, as if this clip interrupted "
        "something far more important. Pick one thing actually visible on "
        "screen and mock it flatly — understatement over hype, ending on a "
        "pointed little dig."
    ),
    "humorous_tech": (
        "Crack a joke like a sleep-deprived programmer on a fifth coffee: map "
        "what is happening on screen onto software life — crashes, updates, "
        "lag, endless loading bars. Keep the real scene clearly recognizable "
        "underneath the joke."
    ),
    "humorous_non_tech": (
        "Joke like the funny uncle at a family dinner: one warm, punchy "
        "comparison to food, chores, weather, pets, or naps, in about "
        "20-30 words — not a rambling story. Use zero technology words "
        "and zero niche references — plain enough that grandmother "
        "laughs too."
    ),
}


def build_qwen_direct_prompt(style: str) -> str:
    persona = QWEN_DIRECT_PERSONAS.get(
        style, f'Write one English caption for this clip in a clear "{style}" voice.')
    return persona + QWEN_DIRECT_FORMAT_RULES


CAPTION_TAG_RE = re.compile(r"<caption_output>\s*(.*?)\s*</caption_output>",
                            re.DOTALL | re.IGNORECASE)
# Truncation tolerance: opening tag present but the reply was cut off before
# the closing tag — take everything after the opening tag, minus any partial
# closing tag fragment at the end.
CAPTION_TAG_OPEN_RE = re.compile(r"<caption_output>\s*(.*)$", re.DOTALL | re.IGNORECASE)


def extract_caption_tag(raw: str) -> str:
    """Caption text from a qwen_direct reply, or "" when no tag is present."""
    if not raw:
        return ""
    m = CAPTION_TAG_RE.search(raw)
    if m:
        return m.group(1).strip()
    m = CAPTION_TAG_OPEN_RE.search(raw)
    if m:
        return re.sub(r"</?caption[^>]*$", "", m.group(1)).strip()
    return ""