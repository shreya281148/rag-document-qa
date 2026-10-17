from ragqa.prompts import NOT_FOUND, SYSTEM_PROMPT, build_messages, parse_user_message
from ragqa.vectorstore import Hit


def make_hits():
    return [
        Hit("a", "handbook.md", 1, 0, "Leave is 28 days.\nCarry over up to 5 days.", 0.8),
        Hit("b", "policy.pdf", 3, 2, "Passwords expire after 180 days.", 0.7),
    ]


def test_messages_have_system_rules_numbered_context_and_question():
    msgs = build_messages("How much leave?", make_hits())
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    user = msgs[1]["content"]
    assert "[1] (handbook.md, p.1)" in user and "[2] (policy.pdf, p.3)" in user
    assert user.rstrip().endswith("QUESTION: How much leave?")
    assert NOT_FOUND in SYSTEM_PROMPT  # the model is given an explicit way out


def test_build_then_parse_roundtrip():
    question, passages = parse_user_message(build_messages("How much leave?", make_hits())[1]["content"])
    assert question == "How much leave?"
    assert passages == [(1, "Leave is 28 days.\nCarry over up to 5 days."), (2, "Passwords expire after 180 days.")]
