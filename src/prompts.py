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
    "formal": (20, 40),
    "sarcastic": (12, 30),
    "humorous_tech": (12, 30),
    "humorous_non_tech": (12, 30),
}
DEFAULT_WORD_RANGE = (12, 30)


# V2 is emoji-free. strip_emojis is the code-level guarantee that no emoji
# reaches the final output even if a model ignores the "No emoji" instruction.
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF"
    "\U0001F1E6-\U0001F1FF\U0001F900-\U0001F9FF\U0000FE00-\U0000FE0F"
    "\U0000200D\U00002139\U00002194-\U000021AA\U00002300-\U000023FF]",
    flags=re.UNICODE,
)


def strip_emojis(text: str) -> str:
    """Remove any emoji/pictographs and tidy the spacing they leave behind."""
    if not text:
        return text
    cleaned = _EMOJI_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    return cleaned


# Deterministic per-clip rotation for the humorous_non_tech opening.
#
# Why this exists: the style prompt used to merely SUGGEST four openings, and
# the writer model collapsed onto the first one — 11 of 12 clips in the gcs12
# dry run opened with "POV:", and "When you" / "Me trying to" were never used
# once. Across a 12-clip judged batch that reads as one formula on repeat.
# Suggestions don't fix it (the model just re-picks its favourite), and the
# tasks run in parallel threads so they can't see each other's choices — so
# the angle is ASSIGNED per clip from its position in the task list, giving a
# perfectly even round-robin with no shared state and no randomness.
NON_TECH_OPENINGS = [
    ('POV:', 'drop the viewer inside the moment, second person'),
    ('When you', 'frame it as a moment everyone recognises'),
    ('Me trying to', 'self-deprecating, first person'),
    ('That feeling when', 'name the shared emotion behind the moment'),
]


def non_tech_opening_rule(variety_index: int) -> str:
    """The assigned opening for this clip's humorous_non_tech caption, phrased
    so the model can't mistake the instruction for text to copy — an earlier
    wording ("Open with ...") got echoed literally into two captions as
    "Opening with: ..." / "Opening phrase: ...")."""
    phrase, why = NON_TECH_OPENINGS[variety_index % len(NON_TECH_OPENINGS)]
    return (f'the caption text itself must begin with the exact words "{phrase}" '
            f'({why})')


# Per-clip emoji palettes, rotated the same way and for the same reason.
#
# Giving every clip one long menu of "good" emoji just moves the collapse:
# handed a list, the writer takes whatever sits first (👏 landed on 8 of 12
# clips, 🔄 on 9 of 12). Rotating a SMALL palette per clip keeps the choice
# genuine — the model still picks whichever of two or three actually fits the
# joke — while guaranteeing the batch doesn't converge on one emoji.
#
# The palettes are organised by what the emoji has to DO for that style:
#   sarcastic         -> mark the irony (mock-celebrate the mundane), or none
#   humorous_tech     -> reinforce the tech frame that defines the style
#   humorous_non_tech -> carry the human feeling; never a device
EMOJI_PALETTES = {
    "sarcastic": [
        "no emoji at all — end the line bare, the deadpan is the joke",
        "👏 or 🎉 (mock-applause for something utterly mundane)",
        "🏆 or 🥇 (mock-award for a non-achievement)",
        "🥱 or ⭐ (mock-boredom / sarcastic gold star)",
    ],
    "humorous_tech": [
        "💻 or 🖥️ (the machine itself)",
        "🐛 or ⚠️ (bug / error framing)",
        "🔌 or 🔋 (power / hardware framing)",
        "🔄 or ⚙️ (loop / process framing)",
    ],
    "humorous_non_tech": [
        "😅 (nervous, caught-out laughter)",
        "🥲 (fond suffering)",
        "🙃 (resigned absurdity)",
        "😭 or ☕ (dramatic despair, or a mundane everyday prop)",
    ],
}


def emoji_palette_rule(style: str, variety_index: int) -> str:
    """This clip's assigned emoji palette for one style, or "" if the style
    has none (formal never carries an emoji)."""
    # V2: personas are emoji-free — no emoji is injected into any style.
    return ""


def stable_variety_index(key: str) -> int:
    """Fallback when there's no task position to rotate on (e.g. the web
    demo runs one clip at a time). Python's built-in hash() is salted per
    process, so use a stable digest instead — the same clip always gets the
    same angle, which keeps demo output reproducible."""
    return int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % len(NON_TECH_OPENINGS)


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
    r"respawn|spawn(?:s|ed|ing)?|npc|hitbox|speed-?run(?:ning|s)?|pathfinding|"
    r"tetris|minecraft|app|gadget|robot(?:ic)?s?"
    r")\b",
    re.IGNORECASE,
)


def has_tech_jargon(text: str) -> bool:
    """True if a caption contains a term that would break humorous_non_tech's
    'no technical jargon' definition."""
    return bool(TECH_JARGON_RE.search(text))


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
    """Strip prompt-instruction leakage, collapse newlines, drop stray wrapping
    quotes, and de-duplicate an opener the model repeated after a leaked label.
    Returns a single clean line."""
    cleaned = " ".join(str(text).split())  # collapses newlines and runs of spaces
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
    cleaned = strip_emojis(cleaned)  # V2: emoji-free guarantee
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
        'FORMAL — official definition: "Professional, objective, factual tone." '
        'Persona: a composed, authoritative documentary narrator or museum curator. '
        'Roughly 20-40 words. Describe ONLY what is visibly on screen in precise, '
        'elevated, professional English — measured and objective, with zero emotion, '
        'zero jokes, no slang, and no emoji. Do not open with "The scene shows" or '
        '"The video captures," and do not end on a generic summary. Describe the '
        'setting generically ("an urban street," "a modern office") — never name a '
        'real city, country, or landmark; name a playback effect (time-lapse, slow '
        'motion) only if the report states it, and never name capture equipment '
        '(drone, lens, exposure), and NEVER name the capture technique behind a visible '
        'effect — do not write "long exposure," "long-exposure," "time-lapse," '
        '"timelapse," or "slow shutter"; describe only the visible effect instead '
        '(e.g. "vehicles appear as streaks of light"). Only state what the report '
        'confirms; never escalate '
        'a lowered or downward gaze into "eyes closed" or "asleep."'
    ),
    "sarcastic": (
        'SARCASTIC — official definition: "Dry, ironic, lightly mocking." Persona: a '
        'deadpan play-by-play commentator forced to narrate the most mundane footage '
        'of their career as if it were a dramatic sporting event, then flatly '
        'deflating it. Roughly 12-30 words. Short, flat sentences in '
        'subject-verb-object order. The irony comes from treating something ordinary '
        'and specific in THIS clip with mock-gravity and then undercutting it — never '
        'from hype, exclamation, or Gen-Z slang. Mock something concrete that is '
        'actually on screen; do not invent a state the report does not confirm. No emoji.'
    ),
    "humorous_tech": (
        'HUMOROUS_TECH — official definition: "Funny, with technology or programming '
        'references." Persona: a relentlessly over-optimistic tech-startup founder who '
        'reframes whatever is on screen as a disruptive product, pitch, or opportunity. '
        'Roughly 12-30 words. Pick exactly ONE tech or startup idea (an MVP, scaling, '
        'latency, shipping, a pivot, a seed round) and build the whole joke around it, '
        'tied explicitly to the specific subject or action from the report — a generic '
        'tech pun that could be pasted onto any video is a FAIL. Stay on the visual; do '
        'not drift into talking about your own job. Vary the startup angle from clip to '
        'clip — do not always reach for an MVP or "just shipped"; rotate across '
        'funding rounds, pivots, growth metrics, demo day, tech debt, or scaling. No emoji.'
    ),
    "humorous_non_tech": (
        'HUMOROUS_NON_TECH — official definition: "Funny, everyday humour with no '
        'technical jargon." Persona: a gloriously melodramatic friend who narrates '
        'ordinary moments as if they were high personal drama. Roughly 12-30 words. '
        'Relatable, big-feelings everyday humor grounded in what the clip actually '
        'shows, matched to its domain. ZERO technical content of any kind — no '
        'computing, internet, gaming, or engineering references, and no video-game '
        'names. The joke must be a real human moment, not a restatement of the scene. '
        'No emoji. Begin the caption with the exact assigned opening phrase given below.'
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
Who or what is the main focus? Describe appearance in detail (species, color, size, clothing, expression, distinguishing features).
Report gaze and eyes only as directly visible. While a subject is actively doing a task (typing, reading, eating), lowered or hidden eyes almost always mean the gaze is aimed at that task — only state the eyes are closed if the video shows it unmistakably.

2. ENVIRONMENT
Where does this take place? Describe the setting, surfaces, objects, background elements, weather, and time of day. Describe the TYPE of place generically (e.g. "a multi-lane urban street," "an office with desks and computer monitors"). Do NOT name a specific real-world city, country, neighborhood, or landmark — guessing a location from visual style (architecture, plants, signage style, language on signs) is exactly the kind of unconfirmed claim that must NOT appear here.

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
List 3-5 specific, quirky, or memorable details that make this video unique — including things that only motion reveals (a repeated gesture, a rhythm in the traffic, a sudden stop). These are the best ingredients for humor and captions.

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


CAPTION_GENERATION_SHARED_RULES = """If the video analysis lacks sufficient detail for a style's word count, write a shorter, purely factual caption instead of inventing content to fill the length.

Rules:
- Write every caption in English only, regardless of any language seen or implied in the video.
- Write like a real person posting on social media, NOT like AI or a textbook.
- Use strong, specific verbs that match what's actually happening in THIS video (examples only, do not default to these every time: chase, navigate, glow, speed-run — vary your verb choice based on the actual footage).
- Use alliteration ONLY if it fits naturally and doesn't force inaccurate wording (e.g. "fluffy feline"). Never sacrifice accuracy for wordplay. If nothing fits naturally, skip it.
- No inner double quotes. Use single quotes if needed.
- No questions, no hashtags, no call-to-action, no markdown.
- Before finalizing, count the words in each caption and confirm it fits the required range for that style.

Grounding: Every claim must come from the video analysis report above. Check the RISKS section of the report — do not include anything flagged there as uncertain or unconfirmed. Never name a real-world city, country, or landmark.

ON-SCREEN TEXT — ABSOLUTE RULE: never quote, transcribe, or paraphrase the wording of any sign, label, screen, or clothing text in a caption, even if the report mentions text exists. Automated OCR of compressed video misreads text confidently, and a caption quoting misread text is scored as a factual error, while a generic description ("a building sign," "text on the screen") is never penalized. Always describe text generically.

BANNED WORDS (never use — they sound like AI):
bustling, captivating, showcases, delves, vibrant, tapestry, multifaceted, realm

EMOJI — the emoji serves the STYLE, not the subject. The clip's content is already in your words; an emoji that just names the subject (🐕 for a dog clip, 🌊 for a beach clip) is decoration that does nothing for the tone being judged. Each style's emoji has a different job: formal carries none ever; sarcastic marks the irony; humorous_tech reinforces the tech frame that defines it; humorous_non_tech carries the human feeling and never a device. Each style is given an assigned palette for this clip below — stay inside it, never copy the emoji from the example captions below, and never end two captions of this clip on the same emoji.

WRONG vs RIGHT FOR SARCASTIC (Dry and Deadpan, Not Hype Slang):
❌ "The kitten struts in dripping with main character energy, aura points maxed, we are so cooked 😭" — loud hype slang, not dry
✅ "The kitten struts past like it owns the place. It does not." — flat, deadpan, ironic; no emoji needed
❌ "Traffic is absolutely giving main character syndrome right now, no cap 💀" — hype-speak, not mockery
✅ "Traffic barely moves. Everyone still drives like they're late for something important. 👏" — dry observation, mock-applause marks the irony

WRONG vs RIGHT FOR HUMOROUS_NON_TECH (Specific and Relatable, Not a Scene Summary):
❌ "POV: When the feline navigates the green garden foliage" — just re-describes the scene, no joke
✅ "POV: You open a bag of snacks as quietly as possible, but the local furry overlord still hears it from a mile away 🍗"
❌ "Me trying to see the beautiful landscape and mountains in the quiet nature" — generic, no relatable angle
✅ "That feeling when you escape to nature for some peace, but the absolute silence starts making you feel highly suspicious 🌲"
"""


def build_caption_generation_prompt(scene_report: str, styles: list, variety_index: int = 0) -> str:
    """Stage 2 prompt: text-only, works purely from the Stage 1 Scene Report
    (no video re-attached — cheaper and faster than a second video call).

    `variety_index` is the clip's position in the task list; it selects this
    clip's assigned humorous_non_tech opening so a whole judged batch doesn't
    come back as twelve "POV:" captions (see NON_TECH_OPENINGS)."""
    style_blocks = "\n\n".join(STYLE_RULES.get(s, f"{s.upper()} (12-30 words): write in this requested style.") for s in styles)
    keys_example = ", ".join(f'"{s}": "..."' for s in styles)

    # Per-clip assignments (opening + emoji palettes). These are rotated by
    # position in the task list precisely because open-ended menus collapse:
    # left to its own devices the writer opened 11 of 12 clips with "POV:" and
    # ended 8 of 12 sarcastic captions on 👏.
    lines = []
    if "humorous_non_tech" in styles:
        lines.append(f"- humorous_non_tech opening: {non_tech_opening_rule(variety_index)}")
    for s in styles:
        palette = emoji_palette_rule(s, variety_index)
        if palette:
            lines.append(f"- {s} emoji: {palette}")
    assignment = ""
    if lines:
        assignment = (
            "\nPER-CLIP ASSIGNMENTS (these rotate from clip to clip so the batch "
            "never repeats one formula — follow them exactly):\n"
            + "\n".join(lines)
            + "\n"
        )
    return f"""Using ONLY the following video analysis report, generate {len(styles)} caption(s).

--- VIDEO ANALYSIS REPORT ---
{scene_report}
--- END REPORT ---

{CAPTION_GENERATION_SHARED_RULES}

Styles required for this clip:

{style_blocks}
{assignment}
Output JSON only with exactly these keys: {{{keys_example}}}
"""


JUDGE_POLISH_SYSTEM_PROMPT = (
    "You punch up captions for humor and technical/sarcastic wit, without "
    "changing the underlying facts. You are given the video's scene details "
    "and a draft caption. Rewrite it to be funnier and sharper in the same "
    "style, but keep it grounded in the same concrete details — do not "
    "invent new facts about the video, and do not make it longer than the "
    "original by more than ~30%. Return only the rewritten caption text, "
    "no preamble."
)


def build_judge_polish_prompt(style: str, scene_hint: str, draft_caption: str) -> str:
    # Pass the full per-style rules (word count, structure, emoji) — not just
    # the one-line description — so a polish pass can't drift the caption out
    # of the structural constraints Stage 2 wrote it under.
    rules = STYLE_RULES.get(style, STYLE_DESCRIPTIONS.get(style, ""))
    return (
        f"Style rules the rewritten caption MUST still satisfy:\n{rules}\n\n"
        f"Scene details: {scene_hint}\n"
        f"Draft caption: {draft_caption}\n\n"
        "Rewrite it now, sharper and funnier, same style, same facts."
    )


PICK_BEST_SYSTEM_PROMPT = (
    "You are the judge for a video-captioning contest. You are given the "
    "video's scene details and several numbered candidate captions for ONE "
    "requested style. The contest scores exactly two things, equally "
    "weighted: (a) accuracy — the caption faithfully reflects the described "
    "scene with no invented facts (inventing on-screen text wording or a "
    "real-world location is an automatic disqualifier), and (b) style "
    "match — it genuinely lands in the officially defined style rather "
    "than being generic or AI-sounding. For the sarcastic and humorous "
    "styles, a genuinely sharp, funny, specific caption beats a safe "
    "generic one — but never at the cost of accuracy. Word counts and "
    "suggested openings are soft guidance, not disqualifiers. Respond with "
    'ONLY a JSON object: {"best": <1-based candidate number>}.'
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
    "is an automatic fail), and (b) style match — does it genuinely land in "
    "the officially defined style; for sarcastic/humorous styles that means "
    "it is actually funny or biting about THIS clip, not generic filler. "
    "Do not deduct points for word count or opening phrasing. Respond with "
    "ONLY a JSON object: {\"score\": <0-10 number>, \"feedback\": \"<one "
    "short actionable sentence for how to improve it if score < 8>\"}."
)


def build_judge_prompt(style: str, scene_hint: str, caption: str) -> str:
    rules = STYLE_RULES.get(style, STYLE_DESCRIPTIONS.get(style, ""))
    return (
        f"Requested style: {style}\nStyle rules: {rules}\n"
        f"Scene details: {scene_hint}\n"
        f"Caption to judge: {caption}"
    )