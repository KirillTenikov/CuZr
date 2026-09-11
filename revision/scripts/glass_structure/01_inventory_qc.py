#!/usr/bin/env python3
import argparse, tarfile, re
from pathlib import Path
import numpy as np, pandas as pd
from common import *

def minimization_stats(text):
    crit=re.findall(r'Stopping criterion\s*=\s*(.+)',text)
    fmax=re.findall(r'Force max component initial, final\s*=\s*([0-9.eE+\-]+)\s+([0-9.eE+\-]+)',text)
    it=re.findall(r'Iterations, force evaluations\s*=\s*(\d+)\s+(\d+)',text)
    return (crit[-1].strip() if crit else None, float(fmax[-1][1]) if fmax else np.nan, int(it[-1][0]) if it else np.nan)

def main(inp,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True); rows=[]; minrows=[]; branch=[]
    aa=archives(inp)
    if len(aa)!=72: print(f'WARNING: found {len(aa)} archives, expected 72 for full matrix')
    seen=set()
    for arc in aa:
        with tarfile.open(arc,'r:gz') as tf:
            man=read_json(tf,'/manifest.json'); therm=read_json(tf,'/thermo_summary.json')
            d3=parse_lammps_data(read_text(tf,'/03_inherent_fixed_cell.data')); d4=parse_lammps_data(read_text(tf,'/04_inherent_box_relaxed.data'))
            log3=read_text(tf,'/03_inherent_fixed_cell.log');log4=read_text(tf,'/04_inherent_box_relaxed.log')
        run=man['run']; init=man['initial_structure']; pot=potential_label(run['potential_id']); comp=run['composition'];seed=int(run['seed']);N=int(run['natoms'])
        key=(pot,comp,seed)
        if key in seen: raise RuntimeError(f'Duplicate matrix cell {key}')
        seen.add(key)
        st4=therm['stages']['04_inherent_box_relaxed']['columns']; st3=therm['stages']['03_inherent_fixed_cell']['columns']
        counts={t:int((d4['types']==t).sum()) for t in np.unique(d4['types'])}
        mass=sum(d4['masses'][t]*n for t,n in counts.items()); density=mass/d4['volume']*AMU_A3_TO_G_CM3
        rows.append(dict(archive=arc.name,potential=pot,composition=comp,seed=seed,N=N,n_Cu=counts.get(1,0),n_Zr=counts.get(2,0),
            Lx_A=d4['L'][0],Ly_A=d4['L'][1],Lz_A=d4['L'][2],xy_A=d4['tilts'][0],xz_A=d4['tilts'][1],yz_A=d4['tilts'][2],
            volume_A3=d4['volume'],V_per_atom_A3=d4['volume']/N,density_g_cm3=density,density_thermo_g_cm3=st4['Density']['final'],
            P_residual_GPa=st4['Press']['final']*1e-4,Pxx_GPa=st4['Pxx']['final']*1e-4,Pyy_GPa=st4['Pyy']['final']*1e-4,Pzz_GPa=st4['Pzz']['final']*1e-4,
            Pxy_GPa=st4['Pxy']['final']*1e-4,Pxz_GPa=st4['Pxz']['final']*1e-4,Pyz_GPa=st4['Pyz']['final']*1e-4,
            PE_per_atom_eV=st4['PotEng']['final']/N,P_fixedcell_GPa=st3['Press']['final']*1e-4,density_npt_g_cm3=therm['stages']['01_relax_npt']['columns']['Density']['final'],
            git_commit=man['git']['commit'],workflow_version=man.get('workflow_version'),potential_sha256=man.get('potential_sha256'),
            count_match=(counts.get(1,0)==init['n_cu'] and counts.get(2,0)==init['n_zr'])))
        for stage,log in [('03',log3),('04',log4)]:
            crit,fmax,it=minimization_stats(log);minrows.append(dict(potential=pot,composition=comp,seed=seed,stage=stage,criterion=crit,generalized_fmax=fmax,iterations=it))
        assert np.array_equal(d3['ids'],d4['ids']) and np.array_equal(d3['types'],d4['types'])
        ds=d4['frac']-d3['frac'];ds-=np.rint(ds);mag=np.linalg.norm(ds*d4['L'],axis=1)
        branch.append(dict(potential=pot,composition=comp,seed=seed,cell_linear_strain_pct=100*(d4['L'][0]/d3['L'][0]-1),volume_change_pct=100*(d4['volume']/d3['volume']-1),
            nonaffine_rms_A=np.sqrt(np.mean(mag**2)),nonaffine_max_A=mag.max(),n_gt_0p05_A=int((mag>0.05).sum())))
    df=pd.DataFrame(rows);df.to_csv(out/'inventory_72.csv',index=False)
    pd.DataFrame(minrows).to_csv(out/'minimization_qc_72.csv',index=False);pd.DataFrame(branch).to_csv(out/'stage03_to_04_branch_qc.csv',index=False)
    print(df.groupby(['potential','composition']).size().unstack(fill_value=0));print('count checks:',df.count_match.all())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='.');p.add_argument('--output',default='analysis_72');a=p.parse_args();main(a.input,a.output)
