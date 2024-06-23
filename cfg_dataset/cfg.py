import sys
import nltk
import random

from nltk.grammar import Nonterminal
from nltk.parse.generate import generate

class CFG:
  def __init__(self, rules_file: str) -> None:
    with open(rules_file, "r") as f:
      rules_str = f.read()
    self.grammar = nltk.grammar.CFG.fromstring(rules_str)
    self.parser = nltk.ChartParser(self.grammar)

  # NOTE: sample CFG in order
  def sample_det(self, n: int) -> str:
    samples = list(generate(self.grammar, n=n))
    strs = ["".join(s) for s in samples]
    return strs

  # NOTE: sample CFG randomly 
  def sample_rand(self, n: int) -> str:
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

      sentences = []
      for _ in range(n):
          sentence = _generate_random(self.grammar, [start], depth)
          sentences.append(sentence)

      sentences_str = ["".join(s) for s in sentences]
      return sentences_str

  def verify(self, string: str) -> bool:
    string_ = [c for c in string]
    trees = list(self.parser.parse(string_))
    return len(trees) > 0




if __name__ == "__main__":
  cfg = CFG("../configs/cfg/simple4.cfg")

  # strings = cfg.sample_det(n=10)
  strings = cfg.sample_rand(n=10)
  print(strings)

  belonging = [cfg.verify(s) for s in strings]
  print(belonging)

  fake_strings = []
  for i in range(len(strings)):
    test_sent = list(strings[i])
    test_sent[0] = "g"
    fake_strings.append("".join(test_sent))
  print(fake_strings)

  belonging = [cfg.verify(s) for s in fake_strings]
  print(belonging)
