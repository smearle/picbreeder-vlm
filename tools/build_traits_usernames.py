#!/usr/bin/env python3
"""Map each generated personality trait to a short, picbreeder-style username.

The traits sweep assigns each agent one second-person personality trait (from
`personality_traits.json`). For the blog's "Top Artists" leaderboard we treat
each distinct trait as a distinct artist/user, so we need a compact handle for
each. This derives a readable CamelCase handle from the salient words of the
trait, with hand-tuned overrides for handles the heuristic gets wrong.

Writes `traits_usernames.json`: { trait_string: username }.
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAITS = REPO / "data" / "personality_traits.json"
OUT = REPO / "data" / "traits_usernames.json"

# Lead-ins stripped from the front (longest first so prefixes match greedily).
LEADINS = [
    r"you are biased towards selecting", r"you superstitiously avoid selecting",
    r"you are biased towards", r"you superstitiously",
    r"you prefer images that look like", r"you prefer images that",
    r"you dislike images that look like", r"you dislike images that",
    r"you are drawn to the texture of", r"you are drawn to",
    r"you are subconsciously searching for", r"you subconsciously",
    r"you are searching for the visual equivalent of",
    r"you are searching for a pattern that looks like",
    r"you are searching for", r"you search for", r"you seek out",
    r"you prioritize images that convey", r"you prioritize",
    r"you are obsessed with", r"you are fascinated by", r"you are repulsed by",
    r"you are afraid of", r"you are nostalgic for", r"you are convinced that",
    r"you are looking for", r"you find comfort in", r"you feel a", r"you feel like",
    r"you believe that publishing", r"you believe that", r"you believe the",
    r"you behave like a", r"you behave like", r"you act like a", r"you act like an",
    r"you act like", r"you act with", r"you treat the", r"you treat",
    r"you have an", r"you have a", r"you get", r"you dislike", r"you prefer the",
    r"you prefer", r"your", r"you",
]
# Filler stripped after the lead-in too.
FILLERS = [
    r"a pattern that looks like", r"a shape that looks like", r"a shape that looks",
    r"a visual representation of", r"a visual", r"the visual equivalent of",
    r"a sense of", r"a portrait of", r"a piece of",
    r"images that look like", r"an image of", r"images of", r"the texture of",
    r"the aesthetic of", r"the smell of", r"the taste of", r"the sound of", r"the way",
    r"that look like", r"that looks like", r"looks like", r"look like",
]
STOP = set("a an the of that this to in on at for with and or but is are be as "
           "its it their your you very most all any some into onto from by like "
           "regardless content selecting images image located".split())
# Hand-tuned handles where the heuristic reads poorly (filled in after review).
OVERRIDES = {}


def slug(trait: str) -> str:
    s = trait.strip().rstrip(".").lower()
    s = re.sub(r"[‘’']", "", s)
    for pat in LEADINS:
        m = re.match(pat + r"\b", s)
        if m:
            s = s[m.end():]
            break
    changed = True
    while changed:
        changed = False
        s = s.strip()
        for pat in FILLERS:
            if s.startswith(pat):
                s = s[len(pat):]; changed = True
    words = [w for w in re.findall(r"[a-z0-9]+", s) if w not in STOP and len(w) > 1]
    if not words:
        words = re.findall(r"[a-z0-9]+", trait.lower())[-2:] or ["artist"]
    pick = words[:2]
    return "".join(w.capitalize() for w in pick)


def main():
    traits = json.load(open(TRAITS))["traits"]
    used, mapping = {}, {}
    for t in traits:
        name = OVERRIDES.get(t) or slug(t)
        if name in used and used[name] != t:
            i = 2
            while f"{name}{i}" in used:
                i += 1
            name = f"{name}{i}"
        used[name] = t
        mapping[t] = name
    json.dump(mapping, open(OUT, "w"), indent=0, ensure_ascii=False)
    print(f"wrote {len(mapping)} usernames -> {OUT}")


if __name__ == "__main__":
    main()
