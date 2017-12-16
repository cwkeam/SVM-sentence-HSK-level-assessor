list_of_hsk = ["hsks_sentences/hsk1.txt","hsks_sentences/hsk2.txt","hsks_sentences/hsk3.txt","hsks_sentences/hsk4.txt","hsks_sentences/hsk5.txt"]




# with open('data') as f:
#     for line in f:
#         inner_list = [elt.strip() for elt in line.split(',')]
#         # in alternative, if you need to use the file content as numbers
#         # inner_list = [int(elt.strip()) for elt in line.split(',')]
#         list_of_lists.append(inner_list)
#
# new_str = string.replace(our_str, 'World', 'Jackson')

# # file-output.py
# f = open('helloworld.txt','w')
# f.write('hello world')
# f.close()
#
f1 = open ('hsk/1.txt', 'w')
f2 = open ('hsk/2.txt', 'w')
f3 = open ('hsk/3.txt', 'w')
f4 = open ('hsk/4.txt', 'w')
f5 = open ('hsk/5.txt', 'w')

for dir in list_of_hsk:
    with open(dir) as f:
        for line in f:
            new_str = line.replace('，', '')
            new_str = new_str.replace('：', '')
            new_str = new_str.replace('！', '')
            new_str = new_str.replace(' ', '')

            if dir == "hsks_sentences/hsk1.txt":
                f1.write(new_str)
            elif dir == "hsks_sentences/hsk2.txt":
                f2.write(new_str)
            elif dir == "hsks_sentences/hsk3.txt":
                f3.write(new_str)
            elif dir == "hsks_sentences/hsk4.txt":
                f4.write(new_str)
            elif dir == "hsks_sentences/hsk5.txt":
                f5.write(new_str)
