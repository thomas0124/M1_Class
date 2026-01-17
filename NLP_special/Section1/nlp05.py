def n_gram(target, n):
    return [target[idx : idx + n] for idx in range(len(target) - n + 1)]


txt = "I am an NLPer"
word = txt.split()


print(n_gram(txt, 3))

print(n_gram(word, 2))
