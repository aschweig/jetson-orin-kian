"""Convert LaTeX math notation to speakable text."""

import re

GREEK = {
    "alpha": "alpha",
    "beta": "beta",
    "gamma": "gamma",
    "delta": "delta",
    "epsilon": "epsilon",
    "theta": "theta",
    "lambda": "lambda",
    "mu": "mu",
    "pi": "pi",
    "sigma": "sigma",
    "phi": "phi",
    "omega": "omega",
}

FUNCS = {
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "log": "log",
    "ln": "ln",
    "exp": "exp",
}


def extract_braced(s: str, i: int):
    if i >= len(s) or s[i] != "{":
        raise ValueError("expected '{'")
    depth = 0
    start = i + 1
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[start:j], j + 1
    raise ValueError("unmatched brace")


def is_compound(expr: str) -> bool:
    """True if the spoken expression contains binary addition or subtraction.

    A leading negative (e.g. "negative x") is not compound.
    """
    expr = expr.strip()
    if not expr:
        return False
    # " plus " anywhere, or " minus " that isn't at the very start
    if " plus " in expr:
        return True
    # Find " minus " that has something before it (binary, not unary)
    m = re.search(r".+\s+minus\s+", expr)
    return bool(m)


def translate_frac_content(num: str, den: str) -> str:
    num_t = _convert_math(num).strip()
    den_t = _convert_math(den).strip()

    # "over" acts as an implicit "end quantity", so no explicit end needed.
    num_q = f"the quantity {num_t}" if is_compound(num_t) else num_t
    den_q = f"the quantity {den_t}" if is_compound(den_t) else den_t
    return f"{num_q} over {den_q}"


def replace_fracs(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s.startswith(r"\frac", i):
            i += 5
            while i < len(s) and s[i].isspace():
                i += 1
            num, i = extract_braced(s, i)
            while i < len(s) and s[i].isspace():
                i += 1
            den, i = extract_braced(s, i)
            out.append(translate_frac_content(num, den))
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def replace_sqrts(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s.startswith(r"\sqrt", i):
            i += 5
            while i < len(s) and s[i].isspace():
                i += 1
            if i < len(s) and s[i] == "[":
                j = s.find("]", i)
                degree = s[i+1:j].strip() if j != -1 else ""
                i = j + 1 if j != -1 else i
                while i < len(s) and s[i].isspace():
                    i += 1
                body, i = extract_braced(s, i)
                body_t = _convert_math(body)
                if is_compound(body_t):
                    out.append(f"{degree} root of the quantity {body_t} end quantity")
                else:
                    out.append(f"{degree} root of {body_t}")
            else:
                body, i = extract_braced(s, i)
                body_t = _convert_math(body)
                if is_compound(body_t):
                    out.append(f"square root of the quantity {body_t} end quantity")
                else:
                    out.append(f"square root of {body_t}")
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def replace_superscripts(s: str) -> str:
    s = re.sub(r'([A-Za-z0-9]+)\s*\^2\b', r'\1 squared', s)
    s = re.sub(r'([A-Za-z0-9]+)\s*\^3\b', r'\1 cubed', s)
    s = re.sub(r'([A-Za-z0-9]+)\s*\^\{([^{}]+)\}', r'\1 to the \2', s)
    s = re.sub(r'([A-Za-z0-9]+)\s*\^([A-Za-z0-9])', r'\1 to the \2', s)
    return s


def replace_subscripts(s: str) -> str:
    s = re.sub(r'([A-Za-z0-9]+)\s*_\{([^{}]+)\}', r'\1 sub \2', s)
    s = re.sub(r'([A-Za-z0-9]+)\s*_([A-Za-z0-9])', r'\1 sub \2', s)
    return s


def replace_commands(s: str) -> str:
    for k, v in GREEK.items():
        s = s.replace("\\" + k, f" {v} ")
    for k, v in FUNCS.items():
        s = s.replace("\\" + k, f" {v} ")

    s = s.replace(r"\cdot", " times ")
    s = s.replace(r"\times", " times ")
    s = s.replace(r"\div", " divided by ")
    s = s.replace(r"\pm", " plus or minus ")
    s = s.replace(r"\mp", " minus or plus ")
    # Long forms first to avoid prefix collisions (\le vs \leq)
    s = s.replace(r"\leq", " less than or equal to ")
    s = s.replace(r"\geq", " greater than or equal to ")
    s = s.replace(r"\neq", " not equal to ")
    s = re.sub(r"\\le(?![a-z])", " less than or equal to ", s)
    s = re.sub(r"\\ge(?![a-z])", " greater than or equal to ", s)
    s = re.sub(r"\\ne(?![a-z])", " not equal to ", s)
    s = re.sub(r"\\lt(?![a-z])", " is less than ", s)
    s = re.sub(r"\\gt(?![a-z])", " is greater than ", s)
    s = s.replace(r"\approx", " approximately equal to ")
    s = s.replace(r"\propto", " is proportional to ")
    s = s.replace(r"\simeq", " is asymptotically equal to ")
    s = s.replace(r"\asymp", " is asymptotic to ")
    s = re.sub(r"\\sim(?![a-z])", " is approximately ", s)
    s = s.replace(r"\to", " to ")
    s = s.replace(r"\infty", " infinity ")

    # Calculus (use regex to avoid prefix collisions: \int vs \infty, \prod vs \propto)
    s = s.replace(r"\partial", " partial ")
    s = re.sub(r"\\int(?![a-z])", " the integral of ", s)
    s = re.sub(r"\\sum(?![a-z])", " the sum of ", s)
    s = re.sub(r"\\prod(?![a-z])", " the product of ", s)
    s = re.sub(r"\\lim(?![a-z])", " the limit of ", s)

    # Set theory (longer commands first to avoid prefix collisions)
    s = s.replace(r"\subseteq", " is a subset of or equal to ")
    s = s.replace(r"\supseteq", " is a superset of or equal to ")
    s = s.replace(r"\subset", " is a subset of ")
    s = s.replace(r"\supset", " is a superset of ")
    s = s.replace(r"\setminus", " minus ")
    s = s.replace(r"\emptyset", " the empty set ")
    s = s.replace(r"\notin", " not in ")
    s = s.replace(r"\cap", " intersect ")
    s = s.replace(r"\cup", " union ")
    # \in must come after \infty, \int, \intersect etc.
    s = re.sub(r"\\in(?![a-z])", " in ", s)

    # Quantifiers
    s = s.replace(r"\forall", " for all ")
    s = s.replace(r"\exists", " there exists ")

    # Negation prefix: \not\approx, \not\in, etc.
    s = s.replace(r"\not", " not")

    # Factorial: ! after a word character, digit, or close paren/bracket
    s = re.sub(r"(?<=[\w)\]])!", " factorial", s)

    return s


def replace_delimiters(s: str) -> str:
    s = s.replace("(", " open paren ")
    s = s.replace(")", " close paren ")
    s = s.replace("[", " open bracket ")
    s = s.replace("]", " close bracket ")
    return s


def cleanup(s: str) -> str:
    s = s.replace("$", "")
    s = s.replace(r"\left", "")
    s = s.replace(r"\right", "")
    s = s.replace("{", " ")
    s = s.replace("}", " ")
    s = s.replace("\\", " ")
    s = s.replace("+", " plus ")
    # Unary negative: leading minus or minus after (, [, =, or their spoken forms
    s = re.sub(r"(?:^|(?<=[(=[]))\s*-\s*", " negative ", s)
    s = re.sub(r"(open (?:paren|bracket)\s+)-\s*", r"\1negative ", s)
    s = s.replace("-", " minus ")
    s = s.replace("=", " equals ")
    s = re.sub(r"\s+", " ", s)
    # Separate adjacent single lowercase letters that form non-words
    # e.g. "ix" → "i x", "xy" → "x y", but leave "sin", "cos", "the", etc.
    s = re.sub(r"\b([a-z]{2,})\b", _split_variables, s)
    return s.strip()


# Words that should NOT be split into individual letters
_KNOWN_WORDS = {
    # Math functions & keywords
    "sin", "cos", "tan", "log", "ln", "exp", "mod", "gcd", "lcm", "max", "min",
    "integral", "sum", "product", "limit", "partial", "derivative",
    # Speech words generated by our converter
    "over", "the", "quantity", "end", "squared", "cubed", "root", "square",
    "plus", "minus", "times", "divided", "equals", "negative", "factorial",
    "open", "close", "paren", "bracket",
    "less", "than", "greater", "equal", "not", "approximately",
    "or", "to", "of", "by", "for", "all", "there", "exists", "in",
    "intersect", "union", "subset", "superset", "set", "empty",
    "is", "and", "proportional", "approximately", "asymptotically", "asymptotic",
    # Greek
    "alpha", "beta", "gamma", "delta", "epsilon", "theta",
    "lambda", "mu", "pi", "sigma", "phi", "omega", "infinity",
    # Ordinals
    "sub",
}


def _split_variables(m: re.Match) -> str:
    word = m.group(1)
    if word in _KNOWN_WORDS:
        return word
    return " ".join(word)


def _convert_math(s: str) -> str:
    """Convert a single math expression (without delimiters) to speech."""
    s = replace_fracs(s)
    s = replace_sqrts(s)
    s = replace_commands(s)
    s = replace_subscripts(s)
    s = replace_superscripts(s)
    s = replace_delimiters(s)
    s = cleanup(s)
    return s


# Math delimiters: $$...$$, $...$, \[...\], \(...\), \begin{equation}...\end{equation}
_MATH_RE = re.compile(
    r"\$\$(.+?)\$\$"           # $$...$$
    r"|\$(.+?)\$"              # $...$
    r"|\\\\?\[(.+?)\\\\?\]"    # \[...\]
    r"|\\\((.+?)\\\)"          # \(...\)
    r"|\\begin\{(?:equation|align|math)\*?\}(.+?)\\end\{(?:equation|align|math)\*?\}",
    re.DOTALL,
)


def latex_to_speech(s: str) -> str:
    """Convert LaTeX math regions to speakable English, leaving prose untouched."""
    def _replace(m: re.Match) -> str:
        # Pick whichever group matched
        expr = next(g for g in m.groups() if g is not None)
        return _convert_math(expr)
    return _MATH_RE.sub(_replace, s)


if __name__ == "__main__":
    # Run: uv run python -m kian.latex_to_speech
    examples = [
        r"$\frac{x}{y}$",
        r"$\frac{a+b}{c}$",
        r"$\frac{x}{y+1}$",
        r"$\frac{a+b}{c+d}$",
        r"$\left(\frac{a+b}{c}\right)$",
        r"$\sqrt{x^2+y^2}$",
        r"$\alpha_i = \frac{1}{n}$",
        r"$x \pm 5$",
        r"$a \div b$",
        r"$x \approx 3.14$",
        r"\(2x + 3 = 7\)",
    ]

    print("=== Examples ===")
    for ex in examples:
        print(f"  {ex}")
        print(f"  -> {latex_to_speech(ex)}")
        print()

    print("=== Tests ===")
    from tests.test_latex_to_speech import run
    run()
