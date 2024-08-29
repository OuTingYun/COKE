import pandas as pd
import numpy as np 


num_var = 50
num_recipes = 16
seed=20


def rn(x,arc,node_name,folder):
    rn_df = {i:'p'+str(int(i.split("_")[0][1:]))+'_'+'ms'+str(int(i.split("_")[1][1:])-1)+'_'+'m'+str(idx) for idx,i in enumerate(node_name)}
    rn = [([i[0]],i[1])for i in list(rn_df.items())]
    # rn = [([i],'p'+str(int(i.split("_")[0][1:]))+'_'+'ms'+str(int(i.split("_")[1][1:])-1)+'_'+'m'+str(idx)) for idx,i in enumerate(node_name)]
    for i in rn:
        arc['from'] = arc['from'].replace(i[0],i[1])
        arc['to'] = arc['to'].replace(i[0],i[1])


    x = x.rename(columns =rn_df)
    print(x.head())
    x.to_csv(f'{folder}/x_rename.csv',index=False)
    arc.to_csv(f'{folder}/arcs_rename.csv',index=False)

for sd in range(10,21):
    folder = f'data' 
    a = pd.read_csv(f'{folder}/A.csv').drop(columns=['Unnamed: 0']).to_numpy()
    x = pd.read_csv(f'{folder}/X.csv')
    arc = pd.read_csv(f'{folder}/arcs.csv')
    node_name = list(x.columns)
    rn(x,arc,node_name,folder)


