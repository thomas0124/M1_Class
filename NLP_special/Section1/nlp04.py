txt = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."
words = txt.replace(".", "").split()

result = {i+1: word[:1] if i in [0, 4, 5, 6, 7, 8, 14, 15, 18] else word[:2] for i, word in enumerate(words)}

print(result)