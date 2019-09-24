
import pandas as pd

dens = 0.845
upper_bound = 0.5 #%
lower_bound = -0.5 #%

N_F35 = 10
N_V22 = 12
N_AV8B = 15
N_SH60 = 50

v_F35 =  340.7643
v_V22 = 183.7674
v_AV8B = 147.5448
v_SH60 = 39.2867

w_F35 = 13.3
w_V22 = 15.03
w_AV8B = 6.74
w_SH60 = 8.05

disp_tot = 42288.4
weight = 40628.0
out_dict = {}
sum = 0
dfs = []
for n_F35 in range(N_F35+1):
    for n_V22 in range(N_V22+1):
        for n_AV8B in range(N_AV8B+1):
            for n_SH60 in range(N_SH60+1):
                w_veh = n_F35*w_F35 + n_V22*w_V22 + n_AV8B*w_AV8B + n_SH60*w_SH60
                w_fuel = (n_F35*v_F35 + n_V22*v_V22 + n_AV8B*v_AV8B + n_SH60*v_SH60)*dens
                w_tot = weight + w_veh + w_fuel
                perc = 100.0*(w_tot - disp_tot)/disp_tot

                if perc >= lower_bound and perc <= upper_bound:
                    out_dict = {'n_F35': [n_F35], 'n_V22': [n_V22], 'n_AV8B': [n_AV8B], 'n_SH60':[n_SH60], 'w_veh':[w_veh], 'w_fuel': [w_fuel], 'perc': [perc]}
                    tdf = pd.DataFrame(out_dict)
                    dfs.append(tdf)

out_df = pd.concat(dfs, ignore_index = False, axis=0, sort=True)

out_df.to_csv('enumerated_aircraft.csv')
