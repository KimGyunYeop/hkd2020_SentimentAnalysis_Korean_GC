# concat sampled naver, daum movie review + tv(4flix, watcha, kinolights, aniplus), interpark, youtube, samsung lions
import pandas as pd
n_df_neg = pd.read_csv('./data/naver_negSample.txt', sep=',')
n_df_pos = pd.read_csv('./data/naver_posSample.txt', sep=',')
n_df_pos = n_df_pos.replace(',' ,' ')
n_df_pos = n_df_pos.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
n_df_neg = n_df_neg.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
n_df = n_df_neg.append(n_df_pos)
print(len(n_df_neg), len(n_df_pos), len(n_df))
n_df = n_df.rename(columns={'label':'rating', 'reviews' : 'review'})

all_df = n_df
print(len(all_df))
f4_df = pd.read_csv('./data/4flix_Reviews.txt', sep='\t')
f4_df = f4_df.drop(columns=['Unnamed: 0'])
f4_df = f4_df.rename(columns={'label':'rating', 'reviews' : 'review'})
f4_df = f4_df.drop_duplicates(subset = ['review'])
print('f4', len(f4_df))
print(len(f4_df[f4_df['rating']==1]), len(f4_df[f4_df['rating']==0]))
all_df = all_df.append(f4_df)
print(len(all_df))
wc_df = pd.read_csv('./data/watcha_Reviews.txt', sep='\t')
wc_df = wc_df.drop(columns=['Unnamed: 0', 'id'])
wc_df = wc_df.rename(columns={'label':'rating', 'reviews' : 'review'})
wc_df = wc_df.drop_duplicates(subset = ['review'])
print('wc', len(wc_df))
print(len(wc_df[wc_df['rating']==1]), len(wc_df[wc_df['rating']==0]))
all_df = all_df.append(wc_df)
print(len(all_df))
kino_df = pd.read_csv('./data/kinolights_Reviews.txt', sep='\t')
kino_df = kino_df.drop(columns=['Unnamed: 0', 'id'])
kino_df = kino_df.rename(columns={'label':'rating', 'reviews' : 'review'})
kino_df = kino_df.drop_duplicates(subset = ['review'])
print('kino', len(kino_df))
print(len(kino_df[kino_df['rating']==1]), len(kino_df[kino_df['rating']==0]))
all_df = all_df.append(kino_df)
print(len(all_df))
ani_df = pd.read_csv('./data/aniplus_Reviews.txt', sep='\t')
ani_df = ani_df.drop(columns=['Unnamed: 0'])
ani_df =ani_df.rename(columns={'label':'rating', 'reviews' : 'review'})
ani_df = ani_df.drop_duplicates(subset = ['review'])
print('ani', len(ani_df))
print(len(ani_df[ani_df['rating']==1]), len(ani_df[ani_df['rating']==0]))
all_df = all_df.append(ani_df)
print('ani', len(all_df))

sl_df = pd.read_csv('./data/samsungLions_Reviews.txt', sep='\t')
sl_df = sl_df.drop(columns= ['Unnamed: 0', 'date', 'time', 'titles'])
sl_df = sl_df.rename(columns= {'win' : 'rating', 'reviews' : 'review'})
len_ori = len(sl_df)
fp_df = sl_df
sl_df = sl_df.drop_duplicates(subset = ['review'])
# use only negative data
print(len(sl_df))
print(len(sl_df[sl_df['rating']==1]), len(sl_df[sl_df['rating']==0]))
sl_df = sl_df[sl_df['rating']==0]
print('negative', len(sl_df))
sl2_df = pd.read_csv('./data/samsung_reviews_contents_v2.txt', sep='\t')
sl2_df = sl2_df.drop(columns= ['Unnamed: 0'])
sl2_df = sl2_df.rename(columns={'label' : 'rating', 'reviews' : 'review'})
sl2_df = sl2_df[sl2_df['rating']==0]
len_ori += len(sl2_df)
sl2_df = sl2_df.drop_duplicates(subset = ['review'])
fp_df = fp_df.append(sl2_df)
sl3_df = pd.read_csv('./data/samsung_reviews_title_v2.txt', sep='\t')
sl3_df = sl3_df.drop(columns= ['Unnamed: 0'])
sl3_df = sl3_df.rename(columns={'label' : 'rating', 'titles' : 'review'})
sl3_df = sl3_df[sl3_df['rating']==0]
len_ori += len(sl3_df)
sl3_df = sl3_df.drop_duplicates(subset = ['review'])
print('before drop', len_ori)
print('samsung', len(sl_df) + len(sl2_df) + len(sl3_df))
fp_df = fp_df.append(sl3_df)
kh_df = pd.read_csv('./data/kiumHeroes_Reviews_titles.csv', sep='\t')
kh_df = kh_df.drop(columns= ['Unnamed: 0'])
kh_df = kh_df.rename(columns={'labels' : 'rating', 'titles' : 'review'})
print('whole', len(kh_df))
kh_df = kh_df[kh_df['rating']==0]
len_kh = len(kh_df)
kh_df = kh_df.drop_duplicates(subset = ['review'])
print('kh title', len_kh, len(kh_df))
fp_df = fp_df.append(kh_df)
kh2_df = pd.read_csv('./data/kiumHeroes_Reviews_contents.csv', sep='\t')
kh2_df = kh2_df.drop(columns= ['Unnamed: 0'])
kh2_df = kh2_df.rename(columns={'labels' : 'rating', 'reviews' : 'review'})
print('whole', len(kh2_df))
kh2_df = kh2_df[kh2_df['rating']==0]
len_kh += len(kh2_df)
kh2_df = kh2_df.drop_duplicates(subset = ['review'])
print('kh con', len_kh, len(kh2_df))
fp_df = fp_df.append(kh2_df)
nc_df = pd.read_csv('./data/ncDinos_Reviews_titles.csv', sep='\t')
nc_df = nc_df.drop(columns= ['Unnamed: 0'])
nc_df = nc_df.rename(columns={'labels' : 'rating', 'titles' : 'review'})
len_nc = len(nc_df)
nc_df = nc_df.drop_duplicates(subset = ['review'])
print('nc title', len_nc, len(nc_df))
nc2_df = pd.read_csv('./data/ncDinos_Reviews_contents.csv', sep='\t')
nc2_df = nc2_df.drop(columns= ['Unnamed: 0'])
nc2_df = nc2_df.rename(columns={'labels' : 'rating', 'reviews' : 'review'})
len_nc += len(nc_df)
nc2_df = nc2_df.drop_duplicates(subset = ['review'])
print('nc content', len_nc, len(nc2_df))
fp_df = fp_df.append(nc2_df)
fp_sample = fp_df.sample(n=2000)
print(len(fp_sample))
print('fp', len(all_df))
all_df = all_df.append(fp_sample)
print(len(all_df))
list = all_df.values.tolist()
arr = []
idx = 0
ip_f = open("./data/all_ip.txt", encoding='utf-8-sig')
ori_f = open("./data/nsmc/ratings_train.txt", encoding='utf-8-sig')
arr = ori_f.readlines()
arr_ip = ip_f.readlines()
for i in range(len(list)):
    arr.append(str(20000000+len(arr_ip) + idx) + '\t' + str(list[i][0])+ '\t' + str(list[i][1]) + '\n')
    idx+=1
print('ip', len(arr_ip))
arr = arr + arr_ip
print(len(arr))
f = open("./data/final.txt", 'w', encoding='utf-8-sig')
f.writelines(arr)
f.close()


