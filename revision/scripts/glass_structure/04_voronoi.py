#!/usr/bin/env python3
import argparse,tarfile
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
from scipy.spatial import Voronoi,ConvexHull
from common import *
PAD=6.0
def one(path):
    with tarfile.open(path,'r:gz') as tf:man=read_json(tf,'/manifest.json');d=parse_lammps_data(read_text(tf,'/04_inherent_box_relaxed.data'))
    x=d['xyz'];L=d['L'];types=d['types'];N=len(x);offsets=[(0,0,0)]+[(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1) if (i,j,k)!=(0,0,0)];chunks=[];orig=[]
    for oi,o in enumerate(offsets):
        y=x+np.array(o)*L;mask=np.ones(N,bool) if oi==0 else np.all((y>=-PAD)&(y<=L+PAD),axis=1);chunks.append(y[mask]);orig.extend(np.where(mask)[0])
    pts=np.concatenate(chunks);orig=np.asarray(orig,int);v=Voronoi(pts,qhull_options='Qbb Qc Qz');faces=[[] for _ in range(N)];neigh=[[] for _ in range(N)]
    for (a,b),rv in zip(v.ridge_points,v.ridge_vertices):
        if -1 in rv:continue
        if a<N and b<len(pts):faces[a].append(rv);neigh[a].append(orig[b])
        if b<N and a<len(pts):faces[b].append(rv);neigh[b].append(orig[a])
    p=potential_label(man['run']['potential_id']);c=man['run']['composition'];se=int(man['run']['seed']);rows=[];vs=0
    for i in range(N):
        reg=v.regions[v.point_region[i]]
        if not reg or -1 in reg:raise RuntimeError('Unbounded central Voronoi cell; increase PAD')
        vol=ConvexHull(v.vertices[reg]).volume;vs+=vol;cnt=Counter(len(z) for z in faces[i]);nf=len(faces[i]);n3,n4,n5,n6,n7,n8=[cnt.get(k,0) for k in range(3,9)];other=nf-sum((n3,n4,n5,n6,n7,n8))
        rows.append(dict(potential=p,composition=c,seed=se,atom_id=d['ids'][i],center='Cu' if types[i]==1 else 'Zr',vor_volume_A3=vol,nfaces=nf,n3=n3,n4=n4,n5=n5,n6=n6,n7=n7,n8=n8,n_other=other,f5=n5/nf,ico_exact=int(nf==12 and (n3,n4,n5,n6,n7,n8,other)==(0,0,12,0,0,0,0))))
    return dict(potential=p,composition=c,seed=se,box_volume_A3=d['volume'],voro_volume_sum_A3=vs,volume_closure_rel=vs/d['volume']-1),rows
def main(inp,out,workers):
    out=Path(out);allrows=[];qc=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(one,str(p)) for p in archives(inp)]
        for f in as_completed(fs):m,r=f.result();qc.append(m);allrows.extend(r)
    a=pd.DataFrame(allrows);q=pd.DataFrame(qc);a.to_csv(out/'voronoi_atoms_72.csv.gz',index=False,compression='gzip');q.to_csv(out/'voronoi_volume_qc.csv',index=False)
    if q.volume_closure_rel.abs().max()>1e-10:raise RuntimeError('Voronoi volume closure failed; increase PAD')
    rows=[]
    for (p,c,se),g in a.groupby(['potential','composition','seed']):
        row=dict(potential=p,composition=c,seed=se,vor_faces_mean=g.nfaces.mean(),f5_mean=g.f5.mean(),ico_fraction=g.ico_exact.mean())
        for sp in ['Cu','Zr']:
            z=g[g.center==sp];row[f'ico_{sp}_fraction']=z.ico_exact.mean();row[f'f5_{sp}_mean']=z.f5.mean();row[f'faces_{sp}_mean']=z.nfaces.mean();row[f'volume_{sp}_mean_A3']=z.vor_volume_A3.mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'voronoi_72.csv',index=False)
    idx=['n3','n4','n5','n6','n7','n8','n_other'];top=[]
    for (p,c,sp),g in a.groupby(['potential','composition','center']):
        for rank,(ind,count) in enumerate(g.groupby(idx).size().sort_values(ascending=False).head(15).items(),1):
            z=dict(potential=p,composition=c,center=sp,rank=rank,count=int(count),fraction=count/len(g));z.update(dict(zip(idx,ind)));top.append(z)
    pd.DataFrame(top).to_csv(out/'voronoi_top_indices.csv',index=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',default='.');p.add_argument('--output',default='analysis_72');p.add_argument('--workers',type=int,default=2);a=p.parse_args();main(a.input,a.output,a.workers)
