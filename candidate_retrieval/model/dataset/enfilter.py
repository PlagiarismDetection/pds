# All files and directories ending with .txt and that don't begin with a dot:
content=[]

with open('WIKI_science.txt', mode='r', encoding='utf-8') as file:
    lines = file.readlines()
    
    for line in lines:
        if len(line) > 100:
            content.append(line)


with open('eng_wiki_para.txt', mode='a', encoding='utf-8') as file:
    for line in content:
        file.writelines(line)
