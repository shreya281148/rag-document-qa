"""Prompt engineering: how retrieved text and the question are presented to the model.

The prompt does three jobs that reduce hallucination:
  1. It tells the model to answer ONLY from the supplied context.
  2. It gives the model an explicit way out ("I could not find this...") so it is
     not forced to invent something when the documents don't contain the answer.
  3. It numbers every passage so the model can cite [1], [2]... and a human can
     check each claim against the source.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle
    from .vectorstore import Hit

NOT_FOUND = "I could not find this in the provided documents."

SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the numbered context passages provided.\n"
    "The passages are excerpts from the user's documents, which can be of any kind (reports, contracts, "
    "manuals, CVs, invoices, papers). Text extracted from PDFs often has broken lines, and a passage may "
    "start or end mid-sentence.\n"
    "Rules:\n"
    "1. Use only the context. Do not use outside knowledge.\n"
    "2. A question may refer to the subject of the documents with words like \"he\", \"she\", \"it\", "
    "\"they\" or \"this document\". Treat that as referring to the subject of the passages.\n"
    "3. The question may use different words than the passages (for example \"cost\" for \"price\", "
    "\"deadline\" for \"due date\", \"awards\" for \"recognised as\"). If a passage supports an answer in "
    "other words, answer from it.\n"
    "4. If a passage states the answer, even briefly or in a list or table, give it directly. Do not refuse "
    "when the answer is present.\n"
    f"5. Only if no passage contains the answer, reply exactly: {NOT_FOUND}\n"
    "6. Cite the passages you used inline, like [1] or [2].\n"
    "7. Be concise and factual."
)

CONTEXT_HEADER = "CONTEXT:"
QUESTION_HEADER = "QUESTION:"


def build_messages(question: str, hits: list["Hit"]) -> list[dict[str, str]]:
    """Assemble the chat messages sent to the LLM."""
    blocks = [f"[{n}] ({h.source}, p.{h.page})\n{h.text}" for n, h in enumerate(hits, start=1)]
    user = f"{CONTEXT_HEADER}\n" + "\n\n".join(blocks) + f"\n\n{QUESTION_HEADER} {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_PASSAGE = re.compile(r"\[(\d+)\] \([^)\n]*\)\n(.*?)(?=\n\n\[\d+\] \(|\n\n" + QUESTION_HEADER + r")", re.DOTALL)


def parse_user_message(content: str) -> tuple[str, list[tuple[int, str]]]:
    """Inverse of build_messages (used by the offline ExtractiveLLM)."""
    q_at = content.rfind(QUESTION_HEADER)
    question = content[q_at + len(QUESTION_HEADER) :].strip() if q_at != -1 else ""
    passages = [(int(n), text.strip()) for n, text in _PASSAGE.findall(content)]
    return question, passages
