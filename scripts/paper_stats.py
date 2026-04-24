"""Rough word-count / page estimate for the paper draft."""
import re
from pathlib import Path

text = Path("paper/draft.md").read_text(encoding="utf-8")
clean = re.sub(r"[\|\-\*#\[\]\(\)_`]", " ", text)
clean = re.sub(r"\s+", " ", clean)
words = clean.split()
print(f"total words: {len(words)}")
print(f"estimated IEEE 2-col pages (at ~500 body-words/page): {len(words) / 500:.1f}")

sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
print("\nper-section word counts:")
for s in sections[1:]:
    title = s.split("\n")[0][:60]
    body = "\n".join(s.split("\n")[1:])
    w = len(re.sub(r"[\|\-\*\[\]\(\)_`]", " ", body).split())
    print(f"  {title:<55}{w}")
