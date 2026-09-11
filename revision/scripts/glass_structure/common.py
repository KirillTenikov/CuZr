from pathlib import Path
import tarfile, json, re
import numpy as np

AMU_A3_TO_G_CM3 = 1.66053906660
POTENTIAL_ALIASES = {
    '2007_Mendelev-M-I_Cu-Zr_LAMMPS_ipr1': 'EAM2007',
    'EAM_Mendelev_2019_CuZr': 'EAM2019',
    'ACE_514': 'ACE514', 'ACE_1352': 'ACE1352',
    'MACE_A': 'MACE_A', 'MACE_B': 'MACE_B',
    'MACE_C': 'MACE_C', 'MACE_D': 'MACE_D',
}
POTENTIAL_ORDER = ['EAM2007','EAM2019','ACE514','ACE1352','MACE_A','MACE_B','MACE_C','MACE_D']
COMPOSITION_ORDER = ['Cu36Zr64','Cu50Zr50','Cu64Zr36']
SEED_ORDER = [42,43,44]

def archives(input_dir):
    return sorted(Path(input_dir).glob('paper1_*.tar*.gz'))

def member_by_suffix(tf, suffix):
    matches = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f'{tf.name}: expected one *{suffix}, got {len(matches)}')
    return matches[0]

def read_json(tf, suffix):
    return json.load(tf.extractfile(member_by_suffix(tf, suffix)))

def read_text(tf, suffix):
    return tf.extractfile(member_by_suffix(tf, suffix)).read().decode('utf-8','replace')

def parse_lammps_data(text):
    lines = text.splitlines(); natoms = None; ntypes = None; bounds = {}; tilts=(0.0,0.0,0.0); masses={}
    for line in lines[:80]:
        s=line.strip()
        m=re.match(r'^(\d+)\s+atoms$', s)
        if m: natoms=int(m.group(1))
        m=re.match(r'^(\d+)\s+atom types$', s)
        if m: ntypes=int(m.group(1))
        m=re.match(r'^([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s+(xlo xhi|ylo yhi|zlo zhi)$', s)
        if m: bounds[m.group(3)[0]]=(float(m.group(1)),float(m.group(2)))
        m=re.match(r'^([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s+([+\-0-9.eE]+)\s+xy xz yz$', s)
        if m: tilts=tuple(map(float,m.groups()))
    try:
        i=next(i for i,l in enumerate(lines) if l.strip()=='Masses')+1
        while i < len(lines) and not lines[i].strip(): i+=1
        while i < len(lines) and lines[i].strip():
            p=lines[i].split()
            if len(p)>=2 and p[0].isdigit(): masses[int(p[0])]=float(p[1])
            else: break
            i+=1
    except StopIteration:
        pass
    i=next(i for i,l in enumerate(lines) if l.strip().startswith('Atoms'))+1
    while i < len(lines) and not lines[i].strip(): i+=1
    rows=[]
    while i < len(lines) and lines[i].strip():
        p=lines[i].split()
        if len(p)<5 or not p[0].isdigit(): break
        rows.append((int(p[0]),int(p[1]),float(p[2]),float(p[3]),float(p[4])))
        i+=1
    a=np.asarray(rows,float); a=a[np.argsort(a[:,0])]
    if natoms is not None and len(a)!=natoms: raise RuntimeError(f'Parsed {len(a)} atoms but header says {natoms}')
    lo=np.array([bounds[k][0] for k in 'xyz']); L=np.array([bounds[k][1]-bounds[k][0] for k in 'xyz'])
    xyz=(a[:,2:5]-lo)%L
    frac=xyz/L
    return {
        'natoms':len(a), 'ntypes':ntypes, 'ids':a[:,0].astype(int), 'types':a[:,1].astype(int),
        'xyz':xyz, 'frac':frac, 'lo':lo, 'L':L, 'tilts':tilts, 'masses':masses,
        'volume':float(np.prod(L)),
    }

def potential_label(raw_id):
    return POTENTIAL_ALIASES.get(raw_id, raw_id)
