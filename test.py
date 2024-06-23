import nltk
from nltk.parse.generate import generate

simple4 = """
S -> B A | B C A | A C C | B B
A -> F D | F E D | E F | D E
B -> F F | F E
C -> D D | E D | D F D
D -> 'i' 'g' | 'g' 'i' 'i' | 'h' 'g' 'i'
E -> 'g' 'h' | 'h' 'g' | 'g' 'h' 'i'
F -> 'h' 'h' | 'h' 'i' | 'i' 'g' 'i' | 'g' 'g'
"""

grammar = nltk.grammar.CFG.fromstring(simple4)
print(grammar)

parser = nltk.ChartParser(grammar)
print(parser)

test_sentence = "h h i g i".split()

trees = list(parser.parse(test_sentence))

# check if the sentence is valid according to the grammar
if len(trees) > 0:
    print("Sentence is valid!")
else:
    print("Sentence is not valid!")


valid_sentences = list(generate(grammar, n=1))
print(valid_sentences)

trees = list(parser.parse(valid_sentences[0]))
# check if the sentence is valid according to the grammar
if len(trees) > 0:
    print("Sentence is valid!")
else:
    print("Sentence is not valid!")