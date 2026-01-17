txt = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
txt = txt.replace(",", "").replace(".", "")

lst = [len(i) for i in txt.split()]

print(lst)