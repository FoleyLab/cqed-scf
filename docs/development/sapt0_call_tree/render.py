import json, sys
d = json.load(open(sys.argv[1])); full = len(sys.argv) > 2
SKIP = ("<genexpr>", "<listcomp>", "<dictcomp>", "<lambda>")
def show(n, pre="", last=True, top=False):
    nm = n["name"]
    if not top:
        c = f" ×{n['count']}" if n["count"] > 1 else ""
        print(pre + ("└─ " if last else "├─ ") + nm + c)
        pre += "   " if last else "│  "
    kids = [k for k in n["children"] if full or (not any(s in k["name"] for s in SKIP))]
    for i, k in enumerate(kids): show(k, pre, i == len(kids) - 1)
print(d["backend"], d["basis"], d["components"])
show(d["tree"], top=True)
