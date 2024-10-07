import os
def fix(file):
    ok = []
    with open(file, "r") as r:
        text = r.read()
        for line in text.split("\n"):
            if not line: continue
            shit = line.split(" ")
            x = shit[0]
            if "pipetting" in x.lower() or "buretting" in x.lower() or "mov" in x.lower() or (not os.path.isfile(x) and x != "video_path"):
                print(x)
            else:
                ok.append(line)
    with open(file, "w") as w:
        w.write("\n".join(ok))
        
# mov is the issue. pretty sure it doesnt handle mov.
        
fix("train.csv")
print()
fix("test.csv")
print()
fix("valid.csv")
print()