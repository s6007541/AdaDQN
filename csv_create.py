import pandas as pd


corruptions = ["none","gaussian_noise","shot_noise","impulse_noise","speckle_noise","gaussian_blur","defocus_blur","contrast","brightness","saturate"]
# games = ["SpaceInvaders", "StarGunner",'MsPacman','Phoenix','Centipede', 'VideoPinball', 'AirRaid2m_0', 'AirRaid2m_1', 'AirRaid2m_2']
games = ['Phoenix2m_0', 'Phoenix2m_1', 'Phoenix2m_2']
corr_levels = [1,2,3,4,5]
ttas = ['None', 'bn_stats']

for game in games:
    for corr_level in corr_levels:
        returns_ls = []
        for tta in ttas:
            tta_ls = []
            for corr in corruptions: 
                path = f"results/ALE/{game}-v5/{tta}_{corr}_{corr_level}.txt"
                with open(path) as f:
                    lines = f.readlines()
                
                tta_ls.append(lines[1])
            returns_ls.append(tta_ls)

        df = pd.DataFrame(returns_ls, columns= corruptions, index=ttas)
        df.to_csv(f"csv_files/{game}_{corr_level}.csv")
        
    