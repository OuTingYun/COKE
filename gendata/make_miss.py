import pandas as pd
import numpy as np 


import argparse
parser = argparse.ArgumentParser(description='Configuration')
parser.add_argument('--num_recipes', type=int, default=10, help='Number of recipes in dataset')
parser.add_argument('--ratio_of_xfull', type=float, default=0.01, help='Ratio of xfull')
parser.add_argument('--seed', type=int, default=201, help='Seed')
parser.add_argument('--missrate', type=float, default=0.5, help='Missrate')
parser.add_argument('--miss_range', type=float, default=0.02, help='Miss range')
parser.add_argument('--max_steps_per_recipes', type=int, default=15, help='Max steps per recipe')
config = parser.parse_args()

print("="*30)
for k,v in config.__dict__.items():
    print(k,":",v)
print("="*30)

#* Read data
folder = './data' 
a = pd.read_csv(f'{folder}/A.csv').drop(columns=['Unnamed: 0']).to_numpy()
x = pd.read_csv(f'{folder}/x_rename.csv')
arc = pd.read_csv(f'{folder}/arcs_rename.csv')
node_name = list(x.columns)

#* Setting configuration
num_recipes = config.num_recipes
num_miss_recipes = num_recipes -1
num_samples = len(x)
step = len( set([i.split("_")[0] for i in node_name]))
num_node = len(node_name)
X_full_samples = int(num_samples * config.ratio_of_xfull)
X_miss_samples = num_samples - X_full_samples
seed = config.seed
missrate = config.missrate
miss_range = config.miss_range
max_steps_per_recipes = config.max_steps_per_recipes

step_sen_map  = {}
for idx,n in enumerate(node_name):
    group_id = int(n.split('_')[0][1:])
    idx = int(n.split('_')[2][1:])
    if group_id in step_sen_map:
        step_sen_map[group_id].append(idx)
    else:
        step_sen_map[group_id] = [idx]

X_miss_split = []
res = X_miss_samples
for i in range(num_miss_recipes-1):
    if i<30:
        smps = np.random.randint(1,int(res/80),1)
    else:
        smps = np.random.randint(1,res/5,1)
    res -= smps
    X_miss_split.append(smps.item())
X_miss_split.append(res.item())

len(X_miss_split)

x_buf = x.copy()
mr = -1
while (mr>missrate+miss_range or mr<missrate-miss_range):
    seed +=1
    x = x_buf.copy()
    print('seed:',seed,end=", ")
    recipes = {tuple(node_name):list(range(X_full_samples))}
    start_idx = X_full_samples
    np.random.seed(seed)
    x_miss_posi = 0
    while(x_miss_posi<len(X_miss_split)):
        num_smps = X_miss_split[x_miss_posi]
        while True:
            mask_step = np.random.randint(2,size = (1,step)) #1 is keep as 0 is miss
            # mask_step2 = np.random.randint(2,size = (1,step))
            # mask_step3 = np.random.randint(2,size = (1,step))
            # mask_step = np.multiply(mask_step,mask_step2)
            # mask_step = np.multiply(mask_step,mask_step3)
            if  np.sum(mask_step)<max_steps_per_recipes and np.sum(mask_step)>0:
                mask_step = np.where(mask_step>0,1,0)
                break
            else:
                print("refind",num_smps,end='\r')
        mask_idx = []
        for idx,i in enumerate(mask_step[0]):
            if i==1: mask_idx+=step_sen_map[idx]
        # print(sum(mask_idx))
        mask = [ 1 if i in mask_idx else 0 for i in range(len(node_name))]
        new_name = []
        for ni,n in enumerate(mask):
            if n ==1:
                new_name.append(node_name[ni])
        if tuple(new_name) in recipes: 
            print("haha")
            continue
        x_miss_posi+=1
        recipes[tuple(new_name)] = [*range(start_idx,start_idx+num_smps,1)]
        start_idx+=num_smps

    res_dict = recipes.copy()
    len(recipes)

    df = pd.DataFrame(x.iloc[recipes[tuple(node_name)]].copy())
    del recipes[tuple(node_name)]

    idx_rec = { i:k for k,v in recipes.items() for i in v}

    idx_rec.keys()

    for idx,row in x.iterrows():
        if idx<len(df):continue
        r = idx_rec[idx]
        row = x.iloc[idx]
        for col in node_name:
            if col not in r:
                row[col] = np.nan
        df = pd.concat([df, row.to_frame().T])
    mr = round(df.isnull().sum().sum() / (df.shape[0]*df.shape[1]),3)
    print("missrate:",mr)

arc.to_csv(f'{folder}/arcs_miss({round(df.isnull().sum().sum() / (df.shape[0]*df.shape[1]),2)})_rep({num_recipes})_xfull({X_full_samples}).csv',index=False)
df.to_csv(f'{folder}/X_miss({round(df.isnull().sum().sum() / (df.shape[0]*df.shape[1]),2)})_rep({num_recipes})_xfull({X_full_samples}).csv',index=False)
k_l = []
for k,v in recipes.items():
    for i in k:
        k_l.append(i)
if set(k_l)-set(node_name) == set():
    print('Done') 
