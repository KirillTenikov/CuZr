#!/usr/bin/env python3
import argparse,tarfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter,find_peaks
from common import *
DR=0.02;RMAX=10.0;EDGES=np.arange(0,RMAX+DR*.5,DR);R=0.5*(EDGES[:-1]+EDGES[1:]);SHELL=4*np.pi/3*(EDGES[1:]**3-EDGES[:-1]**3);Q=np.arange(.5,15.0001,.02)
def rdfs(types,xyz,L):
    N=len(types);V=np.prod(L);pairs=cKDTree(xyz,boxsize=L).query_pairs(RMAX,output_type='ndarray');d=xyz[pairs[:,1]]-xyz[pairs[:,0]];d-=L*np.rint(d/L);rr=np.sqrt((d*d).sum(1));ta=types[pairs[:,0]];tb=types[pairs[:,1]];ncu=(types==1).sum();nzr=(types==2).sum()
    masks={'total':np.ones(len(rr),bool),'CuCu':(ta==1)&(tb==1),'CuZr':ta!=tb,'ZrZr':(ta==2)&(tb==2)};norm={'total':N*(N-1)/(2*V)*SHELL,'CuCu':ncu*(ncu-1)/(2*V)*SHELL,'CuZr':ncu*nzr/V*SHELL,'ZrZr':nzr*(nzr-1)/(2*V)*SHELL}
    return {k:np.histogram(rr[m],EDGES)[0]/norm[k] for k,m in masks.items()},pairs,rr
def sq(g,rho):
    x=np.pi*R/RMAX;w=np.sin(x)/x;h=(g-1)*w*R*R;qr=Q[:,None]*R[None,:];return 1+4*np.pi*rho*np.sum(np.sin(qr)/qr*h[None,:],axis=1)*DR
def peakmin(g):
    gs=savgol_filter(g,21,3);ix=np.where((R>=2)&(R<=4))[0];pk,_=find_peaks(gs[ix],prominence=.15);ip=ix[pk[np.argmax(gs[ix[pk]])]] if len(pk) else ix[np.argmax(gs[ix])];ix2=np.where((R>R[ip]+.25)&(R<=4.8))[0];mn,_=find_peaks(-gs[ix2],prominence=.03);im=ix2[mn[0]] if len(mn) else ix2[np.argmin(gs[ix2])];return R[ip],gs[ip],R[im],gs[im]
def main(inp,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rr_rows=[];sq_rows=[];cache={}
    for arc in archives(inp):
        with tarfile.open(arc,'r:gz') as tf:man=read_json(tf,'/manifest.json');d=parse_lammps_data(read_text(tf,'/04_inherent_box_relaxed.data'))
        p=potential_label(man['run']['potential_id']);c=man['run']['composition'];se=int(man['run']['seed']);G,pairs,dist=rdfs(d['types'],d['xyz'],d['L']);cache[(p,c,se)]=(d,pairs,dist)
        for pair,g in G.items(): rr_rows.extend(dict(potential=p,composition=c,seed=se,pair=pair,r_A=r,g_r=v) for r,v in zip(R,g))
        S=sq(G['total'],len(d['types'])/d['volume']);sq_rows.extend(dict(potential=p,composition=c,seed=se,q_Ainv=q,S_NN=v) for q,v in zip(Q,S))
    rdf=pd.DataFrame(rr_rows);sf=pd.DataFrame(sq_rows);rdf.to_csv(out/'rdf_72_long.csv.gz',index=False,compression='gzip');sf.to_csv(out/'sq_72_long.csv.gz',index=False,compression='gzip')
    cuts=[]
    for c in COMPOSITION_ORDER:
        for pair in ['CuCu','CuZr','ZrZr']:
            g=rdf[(rdf.composition==c)&(rdf.pair==pair)].groupby('r_A').g_r.mean().to_numpy();rp,hp,rm,hm=peakmin(g);cuts.append(dict(composition=c,pair=pair,ensemble_peak_r_A=rp,ensemble_peak_g=hp,common_cutoff_A=rm,ensemble_min_g=hm))
    cuts=pd.DataFrame(cuts);cuts.to_csv(out/'rdf_common_cutoffs.csv',index=False);cm={(r.composition,r.pair):r.common_cutoff_A for r in cuts.itertuples()}
    cn=[]
    for (p,c,se),(d,pairs,dist) in cache.items():
        t=d['types'];ta=t[pairs[:,0]];tb=t[pairs[:,1]];ncu=(t==1).sum();nzr=(t==2).sum();ncc=np.sum((ta==1)&(tb==1)&(dist<cm[(c,'CuCu')]));ncz=np.sum((ta!=tb)&(dist<cm[(c,'CuZr')]));nzz=np.sum((ta==2)&(tb==2)&(dist<cm[(c,'ZrZr')]));cc=2*ncc/ncu;cz=ncz/ncu;zc=ncz/nzr;zz=2*nzz/nzr
        cn.append(dict(potential=p,composition=c,seed=se,CN_CuCu=cc,CN_CuZr=cz,CN_Cu_total=cc+cz,CN_ZrCu=zc,CN_ZrZr=zz,CN_Zr_total=zc+zz,CN_mean=2*(ncc+ncz+nzz)/len(t),alpha_CuZr=1-(cz/(cc+cz))/(nzr/len(t)),alpha_ZrCu=1-(zc/(zc+zz))/(ncu/len(t))))
    pd.DataFrame(cn).to_csv(out/'coordination_72.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='.');p.add_argument('--output',default='analysis_72');a=p.parse_args();main(a.input,a.output)
