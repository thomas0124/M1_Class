def n_gram(target, n):
    return [target[idx: idx + n] for idx in range(len(target) - n + 1)]

X = n_gram("paraparaparadise", 2)
Y = n_gram("paragraph", 2)


print(f"和集合: {set(X) | set(Y)}")
print(f"積集合: {set(X) & set(Y)}")
print(f"差集合: {set(X) - set(Y)}")
print("se" in (set(X)))
print("se" in (set(Y)))