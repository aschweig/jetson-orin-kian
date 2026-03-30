"""Tests for control phrase matching."""

from kian.app import _match_control


def run():
    tests = [
        # Reset context
        ("Lets start fresh", ("reset", None)),
        ("Reset", ("reset", None)),
        ("Forget everything", ("reset", None)),
        # Change voice
        ("Change your voice", ("new_voice", None)),
        ("New voice", ("new_voice", None)),
        ("Switch voices", ("new_voice", None)),
        # Reset + new voice
        ("Can I talk to someone else", ("reset_and_new_voice", None)),
        ("I want to talk to someone else", ("reset_and_new_voice", None)),
        # Set name
        ("Your name is Sophie", ("set_name", "Sophie")),
        ("I'll call you Max", ("set_name", "Max")),
        ("I will name you Buddy", ("set_name", "Buddy")),
        # Invalid name
        ("Your name is Poopyhead", ("invalid_name", "Poopyhead")),
        # Set grade
        ("I'm in 3rd grade", ("set_grade", "3")),
        ("I'm in grade 3", ("set_grade", "3")),
        ("I am in 3rd grade", ("set_grade", "3")),
        ("I am in grade 3", ("set_grade", "3")),
        ("I am in the 3rd grade", ("set_grade", "3")),
        ("I'm in kindergarten", ("set_grade", "0")),
        ("I'm in pre-K", ("set_grade", "-1")),
        # Grade clamping
        ("I'm in 15th grade", ("set_grade", "12")),
        ("I'm in 0th grade", ("set_grade", "1")),
        # Should NOT match
        ("What is your name", None),
        ("Tell me about grade 3", None),
        ("Hello there", None),
        # Phrases embedded in longer sentences should NOT trigger
        ("Can you reset the conversation for me", None),
        ("I want to forget everything about dogs", None),
        ("Please switch voices if you can", None),
        ("Can I talk to someone else about this", None),
        ("Can we start fresh after this question", None),
        ("I heard you can change your voice", None),
        # Grade/name with wrong subject should NOT trigger
        ("My friend is in grade 3", None),
        ("Her name is Sophie", None),
        # Regex phrases in longer sentences should NOT trigger
        ("Your name is Sophie and you like cats", None),
        ("I am in 3rd grade and I like dogs", None),
        ("I'll call you Buddy after lunch", None),
        ("I'm in kindergarten and I love it", None),
    ]

    failures = []
    for text, expected in tests:
        result = _match_control(text)
        if result != expected:
            failures.append((text, expected, result))

    if failures:
        for text, expected, result in failures:
            print(f"  FAIL: {text!r} -> {result} (expected {expected})")
        raise AssertionError(f"{len(failures)} control phrase test(s) failed")

    print(f"  [{len(tests)} control phrase tests passed]")


if __name__ == "__main__":
    run()
