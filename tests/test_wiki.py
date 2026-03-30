"""Tests for wiki topic extraction and lookup."""

from kian.wiki import extract_topic, strip_qualifiers, WikiLookup

# --- Topic extraction tests ---
TOPIC_CASES = [
    # Basic patterns
    ("what is photosynthesis", "photosynthesis"),
    ("what are prime numbers", "prime numbers"),
    ("tell me about black holes", "black holes"),
    ("tell me more about the solar system", "solar system"),
    ("explain the pythagorean theorem", "pythagorean theorem"),
    ("how does gravity work", "gravity"),
    ("who was George Washington", "George Washington"),
    ("what is the speed of light", "speed of light"),
    ("where is the Sahara Desert", "Sahara Desert"),
    ("why is the sky blue", "sky blue"),
    ("why do cats purr", "cats purr"),

    # Extended patterns
    ("can you tell me about black holes", "black holes"),
    ("can you teach me about electricity", "electricity"),
    ("can you check on the theory of relativity", "theory of relativity"),
    ("I'd like to learn more about black holes", "black holes"),
    ("I want to know about the solar system", "solar system"),
    ("I'm curious about quantum mechanics", "quantum mechanics"),
    ("what do you know about gravity", "gravity"),
    ("what are the details of quantum mechanics", "details of quantum mechanics"),
    ("what happened in World War 2", "World War 2"),
    ("learn more about dinosaurs", "dinosaurs"),

    # Report/essay patterns
    ("I have to write a report about black holes", "black holes"),
    ("I have to do a presentation about gravity", "gravity"),
    ("I need to make a talk about the solar system", "solar system"),
    ("I have to give a presentation on photosynthesis", "photosynthesis"),
    ("I have to write an essay about World War 2", "World War 2"),
    ("I got to do a project on electricity", "electricity"),

    # "X is interesting" patterns
    ("black holes are so cool", "black holes"),
    ("gravity is really interesting", "gravity"),
    ("photosynthesis is pretty important", "photosynthesis"),
    ("World War 2 was crazy", "World War 2"),

    # Should NOT extract a topic
    ("how are you doing today", None),
    ("hi there", None),
    ("thank you very much", None),
    ("can we play a game", None),
    ("tell me a joke", None),
    ("I like pizza", None),
    ("you are so funny", None),
]

# --- Qualifier stripping tests ---
STRIP_CASES = [
    ("history of bamboo", "bamboo"),
    ("basics of electricity", "electricity"),
    ("details of quantum mechanics", "quantum mechanics"),
    ("science of black holes", "black holes"),
    ("origin of the universe", "universe"),
    ("meaning of life", "life"),
    # Should NOT strip
    ("black holes", None),
    ("photosynthesis", None),
    ("gravity", None),
]

# --- Lookup tests (require DB) ---
LOOKUP_CASES = [
    # (query, expected_title_or_None)
    ("what is photosynthesis", "Photosynthesis"),
    ("tell me about black holes", "Black hole"),
    ("tell me more about the solar system", "Solar System"),
    ("how does gravity work", "Gravity"),
    ("what are prime numbers", "Prime number"),
    ("who was George Washington", "George Washington"),
    ("what is the speed of light", "Speed of light"),
    ("what happened in World War 2", "World War II"),
    ("explain the pythagorean theorem", "Pythagorean theorem"),
    ("review the basics of electricity", "Electricity"),
    ("tell me about the history of bamboo", "Bamboo"),
    ("what are the details of quantum mechanics", "Quantum mechanics"),
    # Should not match
    ("how are you doing today", None),
    ("tell me a joke", None),
    ("hi there", None),
]


def run():
    passed = 0
    failed = 0

    print("  Topic extraction:")
    for text, expected in TOPIC_CASES:
        result = extract_topic(text)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"    FAIL  \"{text}\"")
            print(f"      expected: {expected}")
            print(f"      got:      {result}")

    print("  Qualifier stripping:")
    for text, expected in STRIP_CASES:
        result = strip_qualifiers(text)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"    FAIL  \"{text}\"")
            print(f"      expected: {expected}")
            print(f"      got:      {result}")

    print("  Lookup (requires DB):")
    try:
        w = WikiLookup()
        if not w.available:
            print("    SKIP (database not found)")
        else:
            for query, expected_title in LOOKUP_CASES:
                w._retrieved.clear()
                result = w.search(query)
                if result:
                    got_title = result.splitlines()[0].removeprefix("[Reference: ").removesuffix("]")
                else:
                    got_title = None
                if got_title == expected_title:
                    passed += 1
                else:
                    failed += 1
                    print(f"    FAIL  \"{query}\"")
                    print(f"      expected: {expected_title}")
                    print(f"      got:      {got_title}")
    except Exception as e:
        print(f"    ERROR: {e}")

    print()
    print(f"  {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        raise AssertionError(f"{failed} test(s) failed")


if __name__ == "__main__":
    run()
