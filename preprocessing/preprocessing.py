import string
import toml
from fastcoref import FCoref

WHITESPACES = set(string.whitespace)
PRONOUNS = "i, you, he, she, it, we, they, me, you, him, it, us, them, mine, yours, his, hers, its, ours, theirs, my, your, his, her, our, their, this, that, there, then, when, which, who, what"
PRONOUNS = set(PRONOUNS.split(", "))
CONFIG = toml.load("config.toml")["FCoref"]
COREF_MODEL = FCoref(**CONFIG)

def ResolveReferences(text:str, only_pronouns=True) -> str:
    clusters = COREF_MODEL.predict([text])[0].get_clusters(as_strings=False)
    resolved = set()
    changes = {}
    
    for cluster in clusters:
        min_len = 100000000000
        word_ = None
        for st, ed in cluster:
            word = text[st:ed]
            if len(word) < min_len and not (word.lower() in PRONOUNS or st in resolved):
                # pick smallest word that is not already referenced and is not a pronoun
                word_ = word
                min_len = len(word)
                
        if word_:
            for st, ed in cluster:
                if text[st:ed] != word_ and (not (st in resolved)) and text[st:ed].lower().strip() != word_.lower().strip() and (text[st:ed] in PRONOUNS or not only_pronouns):
                    # no need to reference the same word or already referenced words 
                    changes[st] = (ed, word_)
                resolved.add(st)
    result = ""
    i = 0
    while i < len(text):
        if i in changes:
            ed, ref = changes[i]
            result += f"{text[i:ed]}[{ref}]"
            i = ed # shift
        else:
            result += text[i]
            i += 1
    return result