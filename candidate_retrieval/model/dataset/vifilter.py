import glob
from re import L

alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ'

# All files and directories ending with .txt and that don't begin with a dot:
path_list = glob.glob("./vie_wiki_para/*.txt")

with open('vie_wiki_para.txt', mode='a', encoding='utf-8') as output:
    for path in path_list:
        content = []

        with open(path, mode='r', encoding='utf-8') as input:
            lines = input.readlines()
            for line in lines:
                if len(line) > 170:
                    if line[0] in alphabet:
                        if 'ISBN' not in line:
                            if 'trong lịch Gregory' not in line:
                                if ') là một năm' not in line:
                                    content.append(line)

        for line in content:
            output.writelines(line)
    

