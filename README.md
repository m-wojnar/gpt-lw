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

To use the CFG class, you need to provide the path to the file containing the CFG production rules when creating an instance of the class.
Additionally, you can set the random seed for reproducibility, as shown below:

```python
import random
from cfg_dataset.cfg import CFG

random.seed(42)
cfg = CFG('configs/cfg/simple4.cfg')
```
