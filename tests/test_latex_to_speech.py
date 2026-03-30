"""Tests for LaTeX-to-speech conversion."""

from kian.latex_to_speech import latex_to_speech

CASES = [
    # --- Basic arithmetic ---
    (r"$1 + 2$", "1 plus 2"),
    (r"$x - 3$", "x minus 3"),
    (r"$a = b$", "a equals b"),

    # --- Fractions: atomic / atomic ---
    (r"$\frac{x}{y}$", "x over y"),
    (r"$\frac{1}{2}$", "1 over 2"),
    (r"$\frac{1}{n}$", "1 over n"),

    # --- Fractions: compound numerator ("over" implicitly ends quantity) ---
    (r"$\frac{a+b}{2}$", "the quantity a plus b over 2"),
    (r"$\frac{a+b}{c}$", "the quantity a plus b over c"),

    # --- Fractions: compound denominator ---
    (r"$\frac{2}{a+b}$", "2 over the quantity a plus b"),
    (r"$\frac{x}{y+1}$", "x over the quantity y plus 1"),

    # --- Fractions: both compound ---
    (r"$\frac{a+b}{b+c}$", "the quantity a plus b over the quantity b plus c"),
    (r"$\frac{a+b}{c+d}$", "the quantity a plus b over the quantity c plus d"),

    # --- Mixed expression with multiple fractions ---
    (
        r"$1 + \frac{a+b}{2} + \frac{2}{a+b} + \frac{a+b}{b+c}$",
        "1 plus the quantity a plus b over 2"
        " plus 2 over the quantity a plus b"
        " plus the quantity a plus b over the quantity b plus c",
    ),

    # --- Superscripts ---
    (r"$x^2$", "x squared"),
    (r"$x^3$", "x cubed"),
    (r"$x^{10}$", "x to the 10"),
    (r"$x^n$", "x to the n"),

    # --- Subscripts ---
    (r"$x_1$", "x sub 1"),
    (r"$a_{ij}$", "a sub i j"),

    # --- Square roots: simple (no "the quantity") ---
    (r"$\sqrt{x}$", "square root of x"),
    (r"$\sqrt{x^3}$", "square root of x cubed"),

    # --- Square roots: compound (needs "the quantity ... end quantity") ---
    (r"$\sqrt{x^2+y^2}$", "square root of the quantity x squared plus y squared end quantity"),
    (r"$\sqrt{a - b}$", "square root of the quantity a minus b end quantity"),

    # --- Nth roots ---
    (r"$\sqrt[3]{8}$", "3 root of 8"),
    (r"$\sqrt[3]{a+b}$", "3 root of the quantity a plus b end quantity"),

    # --- Greek letters ---
    (r"$\alpha$", "alpha"),
    (r"$\pi$", "pi"),
    (r"$\theta$", "theta"),
    (r"$\alpha_i = \frac{1}{n}$", "alpha sub i equals 1 over n"),

    # --- Trig / log functions ---
    (r"$\sin(x)$", "sin open paren x close paren"),
    (r"$\cos(\theta)$", "cos open paren theta close paren"),
    (r"$\ln(x)$", "ln open paren x close paren"),

    # --- Unary negative vs binary minus ---
    (r"$-x$", "negative x"),
    (r"$\exp(-i \pi)$", "exp open paren negative i pi close paren"),
    (r"$a - b$", "a minus b"),
    (r"$(-1)$", "open paren negative 1 close paren"),

    # --- Operators ---
    (r"$x \times y$", "x times y"),
    (r"$x \cdot y$", "x times y"),
    (r"$a \div b$", "a divided by b"),
    (r"$x \pm 5$", "x plus or minus 5"),
    (r"$x \approx 3.14$", "x approximately equal to 3.14"),
    (r"$a \leq b$", "a less than or equal to b"),
    (r"$a \geq b$", "a greater than or equal to b"),
    (r"$a \neq b$", "a not equal to b"),
    (r"$a \le b$", "a less than or equal to b"),
    (r"$a \ge b$", "a greater than or equal to b"),
    (r"$a \ne b$", "a not equal to b"),
    (r"$a \lt b$", "a is less than b"),
    (r"$a \gt b$", "a is greater than b"),

    # --- Infinity ---
    (r"$x \to \infty$", "x to infinity"),

    # --- Factorial ---
    (r"$n!$", "n factorial"),
    (r"$5!$", "5 factorial"),
    (r"$\frac{n!}{k!(n-k)!}$",
     "n factorial over the quantity k factorial open paren n minus k close paren factorial"),

    # --- Set theory ---
    (r"$A \cap B$", "A intersect B"),
    (r"$A \cup B$", "A union B"),
    (r"$x \in S$", "x in S"),
    (r"$x \notin S$", "x not in S"),
    (r"$A \subset B$", "A is a subset of B"),
    (r"$A \subseteq B$", "A is a subset of or equal to B"),
    (r"$\emptyset$", "the empty set"),
    (r"$A \setminus B$", "A minus B"),

    # --- Quantifiers ---
    (r"$\forall x$", "for all x"),
    (r"$\exists x$", "there exists x"),

    # --- Proportional / asymptotic ---
    (r"$x \propto y$", "x is proportional to y"),

    # --- Calculus ---
    (r"$\int x$", "the integral of x"),
    (r"$\sum x$", "the sum of x"),
    (r"$\prod x$", "the product of x"),
    (r"$\partial x$", "partial x"),

    # --- Euler's formula ---
    (r"$e^{ix} = \cos(x) + i\sin(x)$",
     "e to the i x equals cos open paren x close paren plus i sin open paren x close paren"),

    # --- Negation prefix ---
    (r"$a \not\approx b$", "a not approximately equal to b"),

    # --- \left / \right delimiters ---
    (r"$\left(\frac{a+b}{c}\right)$", "open paren the quantity a plus b over c close paren"),

    # --- \( ... \) delimiters ---
    (r"\(2x + 3 = 7\)", "2x plus 3 equals 7"),

    # --- Nested fractions ---
    (r"$\frac{\frac{1}{2}}{3}$", "1 over 2 over 3"),

    # --- Prose with ! must NOT become factorial ---
    ("Hey there!", "Hey there!"),
    ("That's a big question!", "That's a big question!"),

    # --- Mixed prose and math ---
    ("The answer is $x^2 + 1$, right?", "The answer is x squared plus 1, right?"),

    # --- No LaTeX (passthrough) ---
    ("just plain text", "just plain text"),
    ("The answer is 42.", "The answer is 42."),
]


def run():
    passed = 0
    failed = 0
    for latex, expected in CASES:
        result = latex_to_speech(latex)
        if result == expected:
            passed += 1
            print(f"  OK  {latex}")
        else:
            failed += 1
            print(f"  FAIL  {latex}")
            print(f"    expected: {expected}")
            print(f"    got:      {result}")
    print()
    print(f"{passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        raise AssertionError(f"{failed} test(s) failed")


if __name__ == "__main__":
    run()
