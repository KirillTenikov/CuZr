#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from common import POTENTIAL_ORDER,COMPOSITION_ORDER
def decomp(d,y):
    grand=d[y].mean();ma=d.groupby('potential')[y].mean();mb=d.groupby('composition')[y].mean();mab=d.groupby(['potential','composition'])[y].mean();nr=3;SSA=len(COMPOSITION_ORDER)*nr*((ma-grand)**2).sum();SSB=len(POTENTIAL_ORDER)*nr*((mb-grand)**2).sum();SSAB=sum(nr*(mab.loc[(a,b)]-ma.loc[a]-mb.loc[b]+grand)**2 for a in POTENTIAL_ORDER for b in COMPOSITION_ORDER);z=d.join(mab.rename('_cell'),on=['potential','composition']);SSE=((z[y]-z._cell)**2).sum();SST=((d[y]-grand)**2).sum();return dict(metric=y,potential_pct=100*SSA/SST,composition_pct=100*SSB/SST,interaction_pct=100*SSAB/SST,seed_within_pct=100*SSE/SST)
def main(out):
    out=Path(out);st=pd.read_csv(out/'inventory_72.csv').rename(columns={'density_g_cm3':'density_calc_g_cm3'});cn=pd.read_csv(out/'coordination_72.csv');vo=pd.read_csv(out/'voronoi_72.csv');sf=pd.read_csv(out/'sq_72_long.csv.gz');sp=[]
    for (p,c,se),g in sf.groupby(['potential','composition','seed']):
        q=g.q_Ainv.to_numpy();s=g.S_NN.to_numpy();m=(q>=1.5)&(q<=4.5);i=np.where(m)[0][np.argmax(s[m])];sp.append(dict(potential=p,composition=c,seed=se,q1_Ainv=q[i],S_q1=s[i]))
    d=st[['potential','composition','seed','density_calc_g_cm3','V_per_atom_A3','P_residual_GPa']].merge(cn,on=['potential','composition','seed']).merge(vo,on=['potential','composition','seed']).merge(pd.DataFrame(sp),on=['potential','composition','seed']);d.to_csv(out/'descriptor_table_72.csv',index=False)
    metrics=['density_calc_g_cm3','CN_mean','CN_Cu_total','CN_Zr_total','alpha_CuZr','q1_Ainv','S_q1','vor_faces_mean','f5_mean','ico_fraction','ico_Cu_fraction'];pd.DataFrame([decomp(d,x) for x in metrics]).to_csv(out/'descriptor_variance_decomposition.csv',index=False)
    rep=[]
    for y in metrics:
        g=d.groupby(['potential','composition'])[y].agg(['mean','std']);rep.append(dict(metric=y,mean_within_cell_SD=g['std'].mean(),median_within_cell_SD=g['std'].median(),max_within_cell_SD=g['std'].max(),grand_mean=d[y].mean(),mean_SD_over_grand_pct=100*g['std'].mean()/abs(d[y].mean())))
    pd.DataFrame(rep).to_csv(out/'descriptor_seed_reproducibility.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='analysis_72');a=p.parse_args();main(a.output)
