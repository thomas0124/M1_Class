import random

def shuffle(txt):
    if len(txt) <= 4:
        return txt
    else:
        start = txt[0]
        end = txt[-1]
        other = random.sample(list(txt[1:-1]), len(txt[1:-1]))
        return "".join([start] + other + [end])

txt = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
ans = [shuffle(w) for w in txt.split()]
print(ans)