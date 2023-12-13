import pandas as pd

corruptions = [] # fill this ["none","gaussian_noise","shot_noise","impulse_noise","speckle_noise","gaussian_blur","defocus_blur","contrast","brightness","saturate"]
games = [] # fill this ["SpaceInvaders", "StarGunner",'MsPacman','Phoenix','Centipede', 'VideoPinball', 'AirRaid2m_0', 'AirRaid2m_1', 'AirRaid2m_2']
corr_levels = [1,2,3,4,5]
methods = ['DQN', 'AdaDQN']

for game in games:
    for corr_level in corr_levels:
        returns_ls = []
        for method in methods:
            method_ls = []
            for corr in corruptions: 
                path = f"results/ALE/{game}-v5/{method}_{corr}_{corr_level}.txt"
                with open(path) as f:
                    lines = f.readlines()
                
                method_ls.append(lines[1])
            returns_ls.append(method_ls)

        df = pd.DataFrame(returns_ls, columns= corruptions, index=ttas)
        df.to_csv(f"csv_files/{game}_{corr_level}.csv")
        
    