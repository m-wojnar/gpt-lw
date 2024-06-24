import random
import sys

import nltk
from nltk.grammar import Nonterminal
from nltk.parse.generate import generate


class CFG:
    def __init__(self, rules_file: str) -> None:
        with open(rules_file, "r") as f:
            rules_str = f.read()

        self.grammar = nltk.grammar.CFG.fromstring(rules_str)
        self.parser = nltk.ChartParser(self.grammar)

    # NOTE: sample CFG in order
    def sample_det(self, n: int) -> list:
        return ["".join(s) for s in generate(self.grammar, n=n)]

    # NOTE: sample CFG randomly
    def sample_rand(self, n: int) -> list:
        def _generate_random(grammar, items, depth):
            if depth == 0 or not items:
                return []

            result = []
            for item in items:
                if isinstance(item, Nonterminal):
                    productions = grammar.productions(lhs=item)
                    if productions:
                        production = random.choice(productions)
                        result.extend(_generate_random(grammar, production.rhs(), depth - 1))
                else:
                    result.append(item)
            return result

        start = self.grammar.start()
        depth = sys.maxsize

        return ["".join(_generate_random(self.grammar, [start], depth)) for _ in range(n)]

    def verify(self, string: str) -> bool:
        trees = list(self.parser.parse(list(string)))
        return len(trees) > 0