# Loss weighting in GPT models

This repository...

## Installation

To download the repository and install the required packages, run the following commands:

```bash
git clone git@github.com:m-wojnar/gpt-lw.git
pip install -r requirements.txt
```

## Training

TODO

## Data generation

The CFG class is a Python class that is used to parse Context-Free Grammar (CFG) production rules from a file and create a parser.

The CFG class makes the following assumptions about the format of the file containing the CFG production rules:

- Each production rule is on a separate line.
- The left-hand side of the production rule is separated from the right-hand side with a colon.
- The left-hand side is a single non-terminal symbol.
- The right-hand side is a sequence of terminal and non-terminal symbols separated by spaces.
- The start symbol is the left-hand side of the first production rule.
- The empty string is represented by an empty right-hand side.
- The terminal symbols are lowercase letters.
- The non-terminal symbols are uppercase words.

### Usage

To use the CFG class, you need to provide the path to the file containing the CFG production rules when creating an instance of the class. Here is an example:

```python
import random # for reproducibility
random.seed(42)
from cfg_dataset.cfg import CFG

cfg = CFG('cfg_dataset/configs/simple4.cfg')
```

Once you have an instance of the CFG class, you can use the following methods:

- `verify(string: str) -> bool`: This method verifies if a string is in the language of the CFG. It returns `True` if the string is in the language of the CFG, and `False` otherwise. **Attention! If the CFG is ambiguous, this method may return `False` even if the string is in the language of the CFG!**
- `sample() -> str`: This method generates a random string from the language of the CFG.
- `generate_dataset(n_examples: int) -> list`: This method generates a dataset of random strings from the CFG language. It returns a list of generated strings.

### Example CFG definition

Example of a CFG definition can be found in the `data/example.cfg` file. The file contains production rules with the following symbols:

- Start symbol: `S`.
- Non-terminal symbols: `S`, `A`, `B`, `C`.
- Terminal symbols: `g`, `h`, `i`.
