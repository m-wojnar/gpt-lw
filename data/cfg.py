import os
from collections import defaultdict
from random import choice


class CFG:
    """
    This class is used to parse the CFG production rules from a file and create a parser.

    Assumed CFG format:
      - Each production rule is on a separate line.
      - The left-hand side of the production rule is separated from the right-hand side with a colon.
      - The left-hand side is a single non-terminal symbol.
      - The right-hand side is a sequence of terminal and non-terminal symbols separated by spaces.
      - The start symbol is the left-hand side of the first production rule.
      - The empty string is represented by an empty right-hand side.
      - The terminal symbols are lowercase letters.
      - The non-terminal symbols are uppercase words.

    Parameters
    ----------
    rules_file : str
        The path to the file containing the CFG production rules.
    """

    def __init__(self, rules_file: str) -> None:
        dir = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(dir, 'parser.template')) as f:
            parser_template = f.read()

        with open(rules_file) as f:
            rules = f.readlines()

        fn_template = 'def p_{i}(p):\n    """{p}"""\n    pass\n\n\n'
        fn_rules = [fn_template.format(i=i, p=p.strip()) for i, p in enumerate(rules)]

        with open(os.path.join(dir, 'cfg_parser.py'), 'w') as f:
            f.write(parser_template.replace('<RULES>', ''.join(fn_rules)))

        from .cfg_parser import verify
        self.verify_fn = verify

        self.start = rules[0].split(':')[0].strip()
        self.rules_dict = defaultdict(list)

        for rule in rules:
            lhs, rhs = rule.split(':')
            lhs = lhs.strip()
            rhs = [r.strip() for r in rhs.split()]
            self.rules_dict[lhs].append(rhs)

    def verify(self, string: str) -> bool:
        """
        Verify if a string is in the language of the CFG.

        Parameters
        ----------
        string : str
            The string to verify.

        Returns
        -------
        bool
            True if the string is in the language of the CFG, False otherwise.
        """

        try:
            self.verify_fn(string)
            return True
        except SyntaxError:
            return False

    def sample(self) -> str:
        """
        Generate a random string from the language of the CFG.

        Returns
        -------
        str
            The generated string.
        """

        def sample(lhs: str) -> str:
            if lhs not in self.rules_dict:
                return lhs

            rhs = choice(self.rules_dict[lhs])
            return ''.join([sample(r) for r in rhs])

        return sample(self.start)

    def generate_dataset(self, n_examples: int) -> list:
        """
        Generate a dataset of random strings from the CFG language.

        Parameters
        ----------
        n_examples : int
            The number of examples to generate.

        Returns
        -------
        list
            A list of generated strings.
        """

        return [self.sample() for _ in range(n_examples)]
