import re
from collections import Counter

class SpellCorrector:

    def __init__(self, words):
        super().__init__()
        self.WORDS = words
        # self.N = sum(self.WORDS.values())

    # @staticmethod
    # def tokens(text):
    #     return REGEX_

    def P(self, word):
        return self.WORDS[word] / self.N

    def most_probable(self, words):
        _known = self.known(words)
        if _known:
            return max(_known, key=self.P)
        else:
            return []

    @staticmethod
    def edit_step(word):
        letters = 'aăâbcdđeêghiklmnoôơpqrstjwfuưvxyáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return (e2 for e1 in self.edit_step(word)
                for e2 in self.edit_step(e1))

    def known(self, words):
        return set(w for w in words if w in self.WORDS)

    def edit_candidates(self, word, assume_wrong=False, fast=True):
        if fast:
            ttt = self.known(self.edit_step(word)) or {word}
        else:
            ttt = self.known(self.edit_step(word)) or self.known(self.edits2(word)) or {word}

        ttt = self.known([word]) | ttt
        return list(ttt)