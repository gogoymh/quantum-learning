





path = "C://유민형//개인 연구//quantum learning//sample//case5.xml"
file = open(path, mode='r', encoding='utf-8')

total_lines = len(file.readlines())
#print(total_lines)
del file


DataContent = []

start = False

file = open(path, mode='r', encoding='utf-8')
for i in range(total_lines):
    line = file.readline()
    if "<HeadLine>" in line:
        #print(line)
        HeadLine = line.split("[")[2].split("]")[0]
    elif "<SubHeadLine>" in line:
        #print(line)
        SubHeadLine = line.split("[")[2].split("]")[0]
    elif "<DataContent>" in line or start:
        #print(line)
        DataContent.append(line)
        if "</DataContent>" in line:
            start = False
            break
        else:
            start = True
        continue

Final = []

for paragraph in DataContent[:-1]:
    Paragraph = []
    if "." in paragraph:
        sentences = paragraph.split(".")
        if len(sentences) == 1:
            if "\n" in sentences[0]:
                Final.append(sentences[0].split("\n")[0])
            else:
                Final.append(sentences[0])
            continue
        for s_idx, val in enumerate(sentences):
            if (s_idx+1) < len(sentences):
                if val[0] == " ":
                    Paragraph.append(val[1:]+".")
                else:
                    Paragraph.append(val+".")
            
            else:
                if len(val) != 1:
                    Final.append(Paragraph)
                    if "\n" in val:
                        Final.append(val.split("\n")[0])
                    else:
                        Final.append(val)
                    break
                else:
                    Final.append(Paragraph)
                    break
    else:
        if "\n" in paragraph:
            if len(paragraph.split("\n")[0]) > 0:
                Final.append(paragraph.split("\n")[0])
            else:
                Final.append("")
        else:
            if len(paragraph) > 0:
                Final.append(paragraph)
        continue

Final.append(DataContent[-1].split("]")[0])




print("=" * 20)
if "&" in HeadLine:
    tmp = ""
    for i in HeadLine.split("&"):
        if ";" in i:
            if len(i.split(";")) == 3:
                tmp = tmp + i.split(";")[-1] + " "
            else:
                tmp = tmp + "."
        else:
            tmp = tmp + i
    HeadLine = tmp
print(HeadLine, "\n")

print("=" * 20)
if "&" in SubHeadLine:
    tmp = ""
    for i in HeadLine.split("&"):
        if ";" in i:
            if len(i.split(";")) == 3:
                tmp = tmp + i.split(";")[-1] + " "
            else:
                tmp = tmp + "."
        else:
            tmp = tmp + i
    SubHeadLine = tmp
print(SubHeadLine, "\n")

print("=" * 20)
for idx_i, i in enumerate(Final):
    if isinstance(i, str):
        if i == " ":
            continue
        if "&" in i:
            tmp = ""
            for k in i.split("&"):
                if ";" in k:
                    if len(k.split(";")) == 3:
                        tmp = tmp + k.split(";")[-1] + " "
                    else:
                        tmp = tmp + "."
                else:
                    tmp = tmp + k
            i = tmp
        if idx_i == 0:
            text = i.split("[")[2]
        else:
            text = i
    else:
        for idx_j, j in enumerate(i):
            if "&" in j:
                tmp = ""
                for k in j.split("&"):
                    if ";" in k:
                        if len(k.split(";")) == 3:
                            tmp = tmp + k.split(";")[-1] + " "
                        else:
                            tmp = tmp + "."
                    else:
                        tmp = tmp + k
                j = tmp
            if idx_j == 0:
                if idx_i == 0:
                    text = j.split("[")[2]
                else:
                    text = j
            else:
                if "%" in j:
                    text = text + j
                else:
                    text = text + " " + j
      
    print(text)

















