import re
from typing import Callable, List
from difflib import SequenceMatcher


class TextPreprocessor:
    def __init__(self, steps: List[Callable] = None):
        if steps is None:
            steps = [
                self.clean_spaces,
                self.remove_exact_duplicates,
                self.remove_fuzzy_duplicates,
                self.collapse_empty_lines,
            ]
        self.steps = steps

    def clean_spaces(self, text: str) -> str:
        return re.sub(r"[^\S\r\n]+", " ", text)

    def remove_exact_duplicates(self, text: str) -> str:
        lines = text.split("\n")
        seen = set()
        cleaned = []

        for l in lines:
            ls = l.strip()
            if ls not in seen:
                seen.add(ls)
                cleaned.append(l)

        return "\n".join(cleaned)


    @staticmethod
    def is_similar(a: str, b: str, threshold: float = 0.88) -> bool:
        if not a or not b:
            return False
        return SequenceMatcher(None, a, b).ratio() >= threshold

    def remove_fuzzy_duplicates(self, text: str) -> str:
        lines = text.split("\n")
        cleaned = []

        for line in lines:
            ls = line.strip()

            if cleaned:
                last = cleaned[-1].strip()
                if self.is_similar(ls, last):
                    continue 

            cleaned.append(line)

        return "\n".join(cleaned)

    def collapse_empty_lines(self, text: str) -> str:
        return re.sub(r"\n\s*\n+", "\n\n", text)

    def run(self, text: str) -> str:
        for step in self.steps:
            text = step(text)
        return text.strip()
