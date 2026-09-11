#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from common import POTENTIAL_ORDER,COMPOSITION_ORDER

def decompose(df,y):
    grand=df[y].mean(); ma=df.groupby('potential')[y].mean(); mb=df.groupby('composition')[y].mean(); mab=df.groupby(['potential','composition'])[y].mean();nr=3
    SSA=len(COMPOSITION_ORDER)*nr*((ma-grand)**2).sum();SSB=len(POTENTIAL_ORDER)*nr*((mb-grand)**2).sum();SSAB=0
    for a in POTENTIAL_ORDER:
        for b in COMPOSITION_ORDER:SSAB+=nr*(mab.loc[(a,b)]-ma.loc[a]-mb.loc[b]+grand)**2
    z=df.join(mab.rename('_cell'),on=['potential','composition']);SSE=((z[y]-z._cell)**2).sum();SST=((df[y]-grand)**2).sum()
    return dict(metric=y,potential_pct=100*SSA/SST,composition_pct=100*SSB/SST,interaction_pct=100*SSAB/SST,seed_within_pct=100*SSE/SST)
def main(out):
    out=Path(out);df=pd.read_csv(out/'inventory_72.csv')
    s=df.groupby(['potential','composition']).agg(density_mean=('density_g_cm3','mean'),density_sd=('density_g_cm3','std'),Vpa_mean=('V_per_atom_A3','mean'),Vpa_sd=('V_per_atom_A3','std'),Pres_mean_GPa=('P_residual_GPa','mean'),Pres_sd_GPa=('P_residual_GPa','std')).reset_index();s.to_csv(out/'state_group_summary.csv',index=False)
    pd.DataFrame([decompose(df,'density_g_cm3'),decompose(df,'V_per_atom_A3'),decompose(df,'P_residual_GPa')]).to_csv(out/'state_variance_decomposition.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='analysis_72');a=p.parse_args();main(a.output)
