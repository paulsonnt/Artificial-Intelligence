import math
from collections import Counter

import pandas
import numpy


def main():
    origText = pandas.read_csv('paragraphs.txt', sep=';', header=None)
    text = numpy.asarray(origText)

    tokens = tokenize(text)

    noChars = removeChars(tokens)

    noNums = removeNums(noChars)

    lowerCase = toLowerCase(noNums)

    noStopWords = removeStopWords(lowerCase)

    p = PorterStemmer()
    portStem = [""] * len(noStopWords)
    for x in range(len(noStopWords)):
        output = ''
        word = ''
        for j in noStopWords[x]:
            if j.isalpha():
                word += j.lower()
            else:
                if word:
                    output += p.stem(word, 0, len(word) - 1)
                    word = ''
                output += j.lower()
        portStem[x] = output

    featureVector = frequentWords(portStem)

    print(featureVector)

    tdm = createtdm(portStem, featureVector)

    clusters = fcan(tdm)

    print(clusters)


def tokenize(text):
    tokenized = [''] * len(text)
    for i in range(len(text)):
        pgraph = text[i]
        txt = str.split(pgraph[0])
        uniqList = [txt[0].strip()]
        for j in range(len(txt) - 1):
            txt[j + 1] = txt[j + 1].strip()
            uniqList.append(txt[j + 1])
        tokenized[i] = ' '.join(uniqList)
    return tokenized


def removeChars(text):
    punctuation = '''`~!@#$%^&*()-_+={}[]|:;"'<>,./?'''
    for i in range(len(text)):
        pgraph = text[i]
        for ch in pgraph:
            if ch in punctuation:
                pgraph = pgraph.replace(ch, "")
        text[i] = pgraph
    return text


def removeNums(text):
    numbers = "1234567890"
    num_words = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "eleven", "twelve", "thirteen",
                 "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty",
                 "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"]
    for i in range(len(text)):
        pgraph = text[i]
        txt = str.split(pgraph)
        for ch in pgraph:
            if ch in numbers:
                pgraph = pgraph.replace(ch, "")
        for x in txt:
            if x in num_words:
                pgraph = pgraph.replace(x, "")
        text[i] = pgraph
    return text


def toLowerCase(text):
    for i in range(len(text)):
        text[i] = text[i].lower()
    return text


def removeStopWords(text):
    file = pandas.read_csv('stop_words.txt', header=None)
    stopWords = numpy.asarray(file)
    for i in range(len(text)):
        pgraph = text[i]
        keep = ""
        txt = str.split(pgraph)
        for j in range(len(txt)):
            if txt[j] not in stopWords:
                keep += txt[j] + " "
        text[i] = keep
    return text


def combineStems(text):
    for i in range(len(text)):
        pgraph = text[i]
        txt = str.split(pgraph)
        for j in range(len(txt)):
            total = pgraph.count(txt[j])
            pgraph = pgraph.replace(txt[j], "", total - 1)
        text[i] = pgraph
    return text


def frequentWords(text):
    freq = []
    most_freq = []
    for i in range(len(text)):
        txt = str.split(text[i])
        most_freq += Counter(txt).most_common(10)
    for i in most_freq:
        if i[0] not in freq and i[1] > 1:
            freq.append(i[0])
    return freq


def createtdm(text, vector):
    tdm = [[0 for x in range(len(vector))] for y in range(len(text))]
    for i in range(len(text)):
        txt = str.split(text[i])
        words = Counter(txt).most_common()
        for x in words:
            if x[0] in vector:
                index = vector.index(x[0])
                tdm[i][index] = x[1]
        print(tdm[i])
    return tdm


def fcan(tdm):
    patterns = tdm
    alpha = 1
    rad_max = 13
    clusters = {1: patterns[0]}
    lengths = {1: 1}
    num_clusters = 1
    paragraphs = {1: [0]}

    for i in range(len(patterns) - 1):
        ed = []
        for x in clusters:
            sum_ed = 0
            for p in range(len(patterns[i + 1])):
                sum_ed += (patterns[i + 1][p] - clusters.get(x)[p]) * (patterns[i + 1][p] - clusters.get(x)[p])
            distance = math.sqrt(sum_ed)
            ed.append(distance)
        index = ed.index(min(ed)) + 1
        if min(ed) < rad_max:
            for n in range(len(clusters.get(x))):
                top = ((lengths.get(index) * clusters.get(index)[n]) + (alpha * patterns[i + 1][n]))
                bottom = lengths.get(index) + 1
                clusters.get(index)[n] = top / bottom
            lengths[index] = lengths.get(index) + 1
            paragraphs.get(index).append(i + 1)
        else:
            num_clusters += 1
            clusters[num_clusters] = patterns[i + 1]
            lengths[num_clusters] = 1
            paragraphs[num_clusters] = [i + 1]
    return paragraphs

class PorterStemmer:

    def __init__(self):
        """The main part of the stemming algorithm starts here.
        b is a buffer holding a word to be stemmed. The letters are in b[k0],
        b[k0+1] ... ending at b[k]. In fact k0 = 0 in this demo program. k is
        readjusted downwards as the stemming progresses. Zero termination is
        not in fact used in the algorithm.

        Note that only lower case sequences are stemmed. Forcing to lower case
        should be done before stem(...) is called.
        """

        self.b = ""  # buffer for word to be stemmed
        self.k = 0
        self.k0 = 0
        self.j = 0  # j is a general offset into the string

    def cons(self, i):
        """cons(i) is TRUE <=> b[i] is a consonant."""
        if self.b[i] == 'a' or self.b[i] == 'e' or self.b[i] == 'i' or self.b[i] == 'o' or self.b[i] == 'u':
            return 0
        if self.b[i] == 'y':
            if i == self.k0:
                return 1
            else:
                return (not self.cons(i - 1))
        return 1

    def m(self):
        """m() measures the number of consonant sequences between k0 and j.
        if c is a consonant sequence and v a vowel sequence, and <..>
        indicates arbitrary presence,

           <c><v>       gives 0
           <c>vc<v>     gives 1
           <c>vcvc<v>   gives 2
           <c>vcvcvc<v> gives 3
           ....
        """
        n = 0
        i = self.k0
        while 1:
            if i > self.j:
                return n
            if not self.cons(i):
                break
            i = i + 1
        i = i + 1
        while 1:
            while 1:
                if i > self.j:
                    return n
                if self.cons(i):
                    break
                i = i + 1
            i = i + 1
            n = n + 1
            while 1:
                if i > self.j:
                    return n
                if not self.cons(i):
                    break
                i = i + 1
            i = i + 1

    def vowelinstem(self):
        """vowelinstem() is TRUE <=> k0,...j contains a vowel"""
        for i in range(self.k0, self.j + 1):
            if not self.cons(i):
                return 1
        return 0

    def doublec(self, j):
        """doublec(j) is TRUE <=> j,(j-1) contain a double consonant."""
        if j < (self.k0 + 1):
            return 0
        if (self.b[j] != self.b[j - 1]):
            return 0
        return self.cons(j)

    def cvc(self, i):
        """cvc(i) is TRUE <=> i-2,i-1,i has the form consonant - vowel - consonant
        and also if the second c is not w,x or y. this is used when trying to
        restore an e at the end of a short  e.g.

           cav(e), lov(e), hop(e), crim(e), but
           snow, box, tray.
        """
        if i < (self.k0 + 2) or not self.cons(i) or self.cons(i - 1) or not self.cons(i - 2):
            return 0
        ch = self.b[i]
        if ch == 'w' or ch == 'x' or ch == 'y':
            return 0
        return 1

    def ends(self, s):
        """ends(s) is TRUE <=> k0,...k ends with the string s."""
        length = len(s)
        if s[length - 1] != self.b[self.k]:  # tiny speed-up
            return 0
        if length > (self.k - self.k0 + 1):
            return 0
        if self.b[self.k - length + 1:self.k + 1] != s:
            return 0
        self.j = self.k - length
        return 1

    def setto(self, s):
        """setto(s) sets (j+1),...k to the characters in the string s, readjusting k."""
        length = len(s)
        self.b = self.b[:self.j + 1] + s + self.b[self.j + length + 1:]
        self.k = self.j + length

    def r(self, s):
        """r(s) is used further down."""
        if self.m() > 0:
            self.setto(s)

    def step1ab(self):
        """step1ab() gets rid of plurals and -ed or -ing. e.g.

           caresses  ->  caress
           ponies    ->  poni
           ties      ->  ti
           caress    ->  caress
           cats      ->  cat

           feed      ->  feed
           agreed    ->  agree
           disabled  ->  disable

           matting   ->  mat
           mating    ->  mate
           meeting   ->  meet
           milling   ->  mill
           messing   ->  mess

           meetings  ->  meet
        """
        if self.b[self.k] == 's':
            if self.ends("sses"):
                self.k = self.k - 2
            elif self.ends("ies"):
                self.setto("i")
            elif self.b[self.k - 1] != 's':
                self.k = self.k - 1
        if self.ends("eed"):
            if self.m() > 0:
                self.k = self.k - 1
        elif (self.ends("ed") or self.ends("ing")) and self.vowelinstem():
            self.k = self.j
            if self.ends("at"):
                self.setto("ate")
            elif self.ends("bl"):
                self.setto("ble")
            elif self.ends("iz"):
                self.setto("ize")
            elif self.doublec(self.k):
                self.k = self.k - 1
                ch = self.b[self.k]
                if ch == 'l' or ch == 's' or ch == 'z':
                    self.k = self.k + 1
            elif (self.m() == 1 and self.cvc(self.k)):
                self.setto("e")

    def step1c(self):
        """step1c() turns terminal y to i when there is another vowel in the stem."""
        if (self.ends("y") and self.vowelinstem()):
            self.b = self.b[:self.k] + 'i' + self.b[self.k + 1:]

    def step2(self):
        """step2() maps double suffices to single ones.
        so -ization ( = -ize plus -ation) maps to -ize etc. note that the
        string before the suffix must give m() > 0.
        """
        if self.b[self.k - 1] == 'a':
            if self.ends("ational"):
                self.r("ate")
            elif self.ends("tional"):
                self.r("tion")
        elif self.b[self.k - 1] == 'c':
            if self.ends("enci"):
                self.r("ence")
            elif self.ends("anci"):
                self.r("ance")
        elif self.b[self.k - 1] == 'e':
            if self.ends("izer"):      self.r("ize")
        elif self.b[self.k - 1] == 'l':
            if self.ends("bli"):
                self.r("ble")  # --DEPARTURE--
            # To match the published algorithm, replace this phrase with
            #   if self.ends("abli"):      self.r("able")
            elif self.ends("alli"):
                self.r("al")
            elif self.ends("entli"):
                self.r("ent")
            elif self.ends("eli"):
                self.r("e")
            elif self.ends("ousli"):
                self.r("ous")
        elif self.b[self.k - 1] == 'o':
            if self.ends("ization"):
                self.r("ize")
            elif self.ends("ation"):
                self.r("ate")
            elif self.ends("ator"):
                self.r("ate")
        elif self.b[self.k - 1] == 's':
            if self.ends("alism"):
                self.r("al")
            elif self.ends("iveness"):
                self.r("ive")
            elif self.ends("fulness"):
                self.r("ful")
            elif self.ends("ousness"):
                self.r("ous")
        elif self.b[self.k - 1] == 't':
            if self.ends("aliti"):
                self.r("al")
            elif self.ends("iviti"):
                self.r("ive")
            elif self.ends("biliti"):
                self.r("ble")
        elif self.b[self.k - 1] == 'g':  # --DEPARTURE--
            if self.ends("logi"):      self.r("log")
        # To match the published algorithm, delete this phrase

    def step3(self):
        """step3() dels with -ic-, -full, -ness etc. similar strategy to step2."""
        if self.b[self.k] == 'e':
            if self.ends("icate"):
                self.r("ic")
            elif self.ends("ative"):
                self.r("")
            elif self.ends("alize"):
                self.r("al")
        elif self.b[self.k] == 'i':
            if self.ends("iciti"):     self.r("ic")
        elif self.b[self.k] == 'l':
            if self.ends("ical"):
                self.r("ic")
            elif self.ends("ful"):
                self.r("")
        elif self.b[self.k] == 's':
            if self.ends("ness"):      self.r("")

    def step4(self):
        """step4() takes off -ant, -ence etc., in context <c>vcvc<v>."""
        if self.b[self.k - 1] == 'a':
            if self.ends("al"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'c':
            if self.ends("ance"):
                pass
            elif self.ends("ence"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'e':
            if self.ends("er"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'i':
            if self.ends("ic"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'l':
            if self.ends("able"):
                pass
            elif self.ends("ible"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'n':
            if self.ends("ant"):
                pass
            elif self.ends("ement"):
                pass
            elif self.ends("ment"):
                pass
            elif self.ends("ent"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'o':
            if self.ends("ion") and (self.b[self.j] == 's' or self.b[self.j] == 't'):
                pass
            elif self.ends("ou"):
                pass
            # takes care of -ous
            else:
                return
        elif self.b[self.k - 1] == 's':
            if self.ends("ism"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 't':
            if self.ends("ate"):
                pass
            elif self.ends("iti"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'u':
            if self.ends("ous"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'v':
            if self.ends("ive"):
                pass
            else:
                return
        elif self.b[self.k - 1] == 'z':
            if self.ends("ize"):
                pass
            else:
                return
        else:
            return
        if self.m() > 1:
            self.k = self.j

    def step5(self):
        """step5() removes a final -e if m() > 1, and changes -ll to -l if
        m() > 1.
        """
        self.j = self.k
        if self.b[self.k] == 'e':
            a = self.m()
            if a > 1 or (a == 1 and not self.cvc(self.k - 1)):
                self.k = self.k - 1
        if self.b[self.k] == 'l' and self.doublec(self.k) and self.m() > 1:
            self.k = self.k - 1

    def stem(self, p, i, j):
        """In stem(p,i,j), p is a char pointer, and the string to be stemmed
        is from p[i] to p[j] inclusive. Typically i is zero and j is the
        offset to the last character of a string, (p[j+1] == '\0'). The
        stemmer adjusts the characters p[i] ... p[j] and returns the new
        end-point of the string, k. Stemming never increases word length, so
        i <= k <= j. To turn the stemmer into a module, declare 'stem' as
        extern, and delete the remainder of this file.
        """
        # copy the parameters into statics
        self.b = p
        self.k = j
        self.k0 = i
        if self.k <= self.k0 + 1:
            return self.b  # --DEPARTURE--

        # With this line, strings of length 1 or 2 don't go through the
        # stemming process, although no mention is made of this in the
        # published algorithm. Remove the line to match the published
        # algorithm.

        self.step1ab()
        self.step1c()
        self.step2()
        self.step3()
        self.step4()
        self.step5()
        return self.b[self.k0:self.k + 1]


if __name__ == '__main__':
    main()
