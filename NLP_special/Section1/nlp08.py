def cipher(text):
    text = [chr(219 - ord(w))  if 97 <= ord(w) <= 122 else w for w in text]
    return ''.join(text)

txt = "I am an NLPer"
ans = cipher(txt)

print(ans)

ans = cipher(ans)
print(ans)