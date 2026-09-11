from pathlib import Path
import argparse
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(); ap.add_argument('--output',default='analysis_72'); args=ap.parse_args(); OUT=Path(args.output); FIG=OUT/'figures'; FIG.mkdir(parents=True,exist_ok=True)
order=['EAM2007','EAM2019','ACE514','ACE1352','MACE_A','MACE_B','MACE_C','MACE_D']
comps=['Cu36Zr64','Cu50Zr50','Cu64Zr36']; x=np.arange(len(comps))
d=pd.read_csv(OUT/'descriptor_table_72.csv')
rdf=pd.read_csv(OUT/'rdf_72_long.csv.gz');sq=pd.read_csv(OUT/'sq_72_long.csv.gz')

def scalar_plot(metric,ylabel,name):
 fig,ax=plt.subplots(figsize=(8,5))
 for p in order:
  g=d[d.potential==p].groupby('composition')[metric].agg(['mean','std']).reindex(comps)
  ax.errorbar(x,g['mean'],yerr=g['std'],marker='o',capsize=3,label=p)
 ax.set_xticks(x,comps);ax.set_xlabel('Composition');ax.set_ylabel(ylabel);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(FIG/name,dpi=220);plt.close(fig)

scalar_plot('density_calc_g_cm3',r'Density (g cm$^{-3}$)','density_vs_composition.png')
scalar_plot('CN_mean','Mean first-shell coordination','coordination_vs_composition.png')
scalar_plot('alpha_CuZr',r'Warren-Cowley-like $\alpha_{Cu-Zr}$','chemical_order_vs_composition.png')
scalar_plot('ico_Cu_fraction',r'Cu-centered $\langle0,0,12,0\rangle$ fraction','cu_icosahedra_vs_composition.png')
scalar_plot('f5_mean','Mean fraction of pentagonal Voronoi faces','pentagonal_faces_vs_composition.png')

for c in comps:
 fig,ax=plt.subplots(figsize=(8,5))
 for p in order:
  g=rdf[(rdf.potential==p)&(rdf.composition==c)&(rdf.pair=='total')].groupby('r_A').g_r.mean()
  ax.plot(g.index,g.values,label=p)
 ax.set_xlim(1.8,8.0);ax.set_xlabel(r'$r$ (Angstrom)');ax.set_ylabel(r'$g(r)$');ax.set_title(c);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(FIG/f'rdf_total_{c}.png',dpi=220);plt.close(fig)

 for pair in ['CuCu','CuZr','ZrZr']:
  fig,ax=plt.subplots(figsize=(8,5))
  for p in order:
   g=rdf[(rdf.potential==p)&(rdf.composition==c)&(rdf.pair==pair)].groupby('r_A').g_r.mean()
   ax.plot(g.index,g.values,label=p)
  ax.set_xlim(1.8,6.5);ax.set_xlabel(r'$r$ (Angstrom)');ax.set_ylabel(fr'$g_{{{pair}}}(r)$');ax.set_title(c);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(FIG/f'rdf_{pair}_{c}.png',dpi=220);plt.close(fig)

 fig,ax=plt.subplots(figsize=(8,5))
 for p in order:
  g=sq[(sq.potential==p)&(sq.composition==c)].groupby('q_Ainv').S_NN.mean()
  ax.plot(g.index,g.values,label=p)
 ax.set_xlim(0.8,10);ax.set_xlabel(r'$q$ (Angstrom$^{-1}$)');ax.set_ylabel(r'$S_{NN}(q)$');ax.set_title(c);ax.legend(ncol=2,fontsize=8);fig.tight_layout();fig.savefig(FIG/f'sq_{c}.png',dpi=220);plt.close(fig)
print('wrote',len(list(FIG.glob('*.png'))),'figures to',FIG)
