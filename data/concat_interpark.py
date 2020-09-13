import pandas as pd

f1 = open("../data/sports_auc.txt", encoding='utf-8-sig')
f2 = open("../data/sports_auc_2.txt", encoding='utf-8-sig')
f3 = open("../data/sports_auc_basketball.txt", encoding='utf-8-sig')
f4 = open("../data/sports_auc_football.txt", encoding='utf-8-sig')
f5 = open("../data/sports_auc_football_2.txt", encoding='utf-8-sig')
f6 = open("../data/sports_auc_golf.txt", encoding='utf-8-sig')
f7 = open("../data/sports_interpark_Reviews.txt", encoding='utf-8-sig')
# convert format of sportsReview
def convertFormat(pos, neg, start, idx_idx, f1):
    dk_df = pd.DataFrame(columns = ['review_id', 'review', 'rating'])
    lines = f1.readlines()
    #new_l = ['\t' if lines=='\n' else line for line in lines]

    # concat sentence of same review
    new_lines = []
    print(len(lines))
    s =''
    for i in range(1, len(lines)):
        sp = lines[i].split('\t')
        if sp[-1][0] == '0' or sp[-1][0] == '1' or sp[-1][0] == '2' or sp[-1][0] == '3' or sp[-1][0] == '4' or sp[-1][0] == '5' or sp[-1][0] == '6' or sp[-1][0] == '7' or sp[-1][0] == '8' or sp[-1][0] == '9' or sp[-1][0] == '10' :
            if len(sp[-1]) == 3 or len(sp[-1]) == 4:
                new_lines.append(s + lines[i])
                s = ''
            else:
                lines[i] = lines[i].replace('\n', ' ')
                s+=lines[i]
        else:
            lines[i] = lines[i].replace('\n', ' ')
            s += lines[i]
    print('new', len(new_lines))

    for i in range(1, len(new_lines)):
        sp = new_lines[i].split('\t')
        a = {"review_id": start + idx_idx, "review": sp[2], "rating": int(sp[3])}
        idx_idx += 1
        dk_df = dk_df.append(a, ignore_index=True)
    print(len(dk_df))
    dk_df = dk_df.drop_duplicates(subset = ['review'])
    print(len(dk_df))
    # list to array
    list_ip = dk_df.values.tolist()
    arr_ip = []
    for i in range(len(list_ip)):
        arr_ip.append(str(list_ip[i][0])+'\t'+str(list_ip[i][1])+'\t'+str(list_ip[i][2])+'\n')
    print('interpark length', len(arr_ip))
    pos += len(dk_df[dk_df['rating'] == 1])
    neg += len(dk_df[dk_df['rating'] == 0])
    return pos, neg, idx_idx, arr_ip

f = open("all_ip.txt", 'w', encoding='utf-8-sig')
pos = 0
neg = 0
pos, neg, idx_idx, arr_ip = convertFormat(pos, neg, 20000000, 0, f1)
f1.close()
pos, neg, idx_idx, arr_ip2 = convertFormat(pos, neg, 20000000, idx_idx, f2)
f2.close()
pos, neg, idx_idx, arr_ip3 = convertFormat(pos, neg, 20000000, idx_idx, f3)
f3.close()
pos, neg, idx_idx, arr_ip4 = convertFormat(pos, neg, 20000000, idx_idx, f4)
f4.close()
pos, neg, idx_idx, arr_ip5 = convertFormat(pos, neg, 20000000, idx_idx, f5)
f5.close()
pos, neg, idx_idx, arr_ip6 = convertFormat(pos, neg, 20000000, idx_idx, f6)
f6.close()
pos, neg, idx_idx, arr_ip7 = convertFormat(pos, neg, 20000000, idx_idx, f7)
f7.close()
list = arr_ip + arr_ip2 + arr_ip3 + arr_ip4 + arr_ip5 + arr_ip6 + arr_ip7
print(len(list))
for i in range(len(list)):
    f.write(str(list[i]))
f.close()
# negative, positive
print("positive: ", pos, "negative: ", neg)









