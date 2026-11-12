import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
LOCATIONS = ('COR', 'RIM', 'SUP', 'SUR', 'INT', 'UNKNOWN')
R_KCAL = 8.31446261815324 / 4184.0
NUMBER = re.compile(r'^[+]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$')
MUTATION = re.compile(r'^([ACDEFGHIKLMNPQRSTVWY])([A-Za-z0-9])(-?\d+)([A-Za-z]?)([ACDEFGHIKLMNPQRSTVWY])$')

def exact_number(value):
    s = str(value).strip()
    if not NUMBER.fullmatch(s):
        raise ValueError('not an exact numeric measurement')
    value = float(s)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('not positive finite')
    return value

def ddg(kd_mut, kd_wt, temperature):
    return R_KCAL * temperature * math.log(kd_mut / kd_wt)

def parse_mutations(text):
    tokens = []
    for s in str(text).split(','):
        match = MUTATION.fullmatch(s.strip())
        if not match:
            raise ValueError('unsupported mutation syntax')
        wt, chain, pos, insertion, mut = match.groups()
        if wt == mut:
            raise ValueError('identity substitution')
        tokens.append((wt, chain, int(pos), insertion, mut))
    if not tokens or len({(t[1], t[2], t[3]) for t in tokens}) != len(tokens):
        raise ValueError('empty or repeated site')
    return tokens

def feature_names():
    return ([f'wt_{a}' for a in AA] + [f'mut_{a}' for a in AA] +
            [f'sub_{a}_{b}' for a in AA for b in AA] +
            [f'loc_{s}' for s in LOCATIONS] +
            ['side_1', 'side_2', 'side_unknown', 'mutation_count', 'temperature_K'])

def features(mutations, locations, partner1, partner2, temperature):
    tokens = parse_mutations(mutations)
    locs = str(locations).split(',')
    x = np.zeros(len(feature_names()), dtype=np.float64)
    for i, (wt, chain, _, _, mut) in enumerate(tokens):
        wi, mi = AA.index(wt), AA.index(mut)
        x[wi] += 1
        x[20 + mi] += 1
        x[40 + wi * 20 + mi] += 1
        loc = locs[i].strip() if i < len(locs) else 'UNKNOWN'
        li = LOCATIONS.index(loc) if loc in LOCATIONS else 5
        x[440 + li] += 1
        side = 0 if chain in partner1 else (1 if chain in partner2 else 2)
        x[446 + side] += 1
    x[449] = len(tokens)
    x[450] = temperature
    return x

class UnionFind:
    def __init__(self, keys): self.parent = {x: x for x in keys}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)

def build_groups(raw):
    pdbs = sorted({r['#Pdb'][:4].upper() for r in raw})
    u = UnionFind(pdbs)
    protein_seen = {}
    for row in raw:
        pdb = row['#Pdb'][:4].upper()
        for linked in re.findall(r'\b([0-9][A-Za-z0-9]{3})_[A-Za-z0-9]+_[A-Za-z0-9]+', row.get('Hold_out_proteins', '')):
            linked = linked.upper()
            if linked in u.parent: u.union(pdb, linked)
        for column in ('Protein 1', 'Protein 2'):
            name = ' '.join(row.get(column, '').lower().split())
            if not name or name in ('antibody', 'antigen', 'unknown'): continue
            if name in protein_seen: u.union(pdb, protein_seen[name])
            else: protein_seen[name] = pdb
    return {p: 'group_' + u.find(p) for p in pdbs}

def assign_splits(rows, seed):
    group_ab = defaultdict(bool)
    for row in rows: group_ab[row['group']] |= bool(row['is_antibody'])
    rng = np.random.default_rng(seed)
    assignment = {}
    for ab in (False, True):
        groups = sorted(g for g, flag in group_ab.items() if flag == ab)
        rng.shuffle(groups)
        if len(groups) < 3:
            raise ValueError(f'Not enough groups for antibody stratum {ab}')
        ntest = max(1, round(len(groups) * 0.15))
        nval = max(1, round(len(groups) * 0.15))
        for i, group in enumerate(groups):
            assignment[group] = 'test' if i < ntest else ('validation' if i < ntest+nval else 'train')
    return [assignment[r['group']] for r in rows]

def load_clean(path, split_seed=20261112):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f, delimiter=';'))
    grouping = build_groups(raw)
    rejected = Counter()
    buckets = defaultdict(list)
    for row_id, r in enumerate(raw):
        try:
            km = exact_number(r['Affinity_mut (M)'])
            kw = exact_number(r['Affinity_wt (M)'])
        except (ValueError, KeyError):
            rejected['nonexact_or_missing_affinity'] += 1; continue
        try:
            t = exact_number(r['Temperature'])
            if not 250 <= t <= 350: raise ValueError('temperature out of range')
        except (ValueError, KeyError):
            rejected['nonexact_assumed_or_missing_temperature'] += 1; continue
        try:
            muts = parse_mutations(r['Mutation(s)_cleaned'])
            parts = r['#Pdb'].split('_')
            if len(parts) != 3: raise ValueError('partner IDs')
            locs = r['iMutation_Location(s)'].split(',')
            if len(locs) != len(muts): locs = ['UNKNOWN'] * len(muts)
            ordered = sorted(zip(muts, locs), key=lambda q: (q[0][1], q[0][2], q[0][3]))
            mutation = ','.join(f'{a}{c}{pos}{ins}{b}' for (a,c,pos,ins,b), loc in ordered)
            location = ','.join(loc for m, loc in ordered)
            x = features(mutation, location, parts[1], parts[2], t)
        except (ValueError, KeyError):
            rejected['unsupported_mutation_or_partner'] += 1; continue
        target = ddg(km, kw, t)
        if not math.isfinite(target):
            rejected['nonfinite_target'] += 1; continue
        record = dict(complex=r['#Pdb'], pdb=parts[0].upper(), mutation=mutation,
                      locations=location, partner1=parts[1], partner2=parts[2], temperature_K=t,
                      ddg_kcal_mol=target, group=grouping[parts[0].upper()],
                      is_antibody=('AB/AG' in r.get('Hold_out_type','')),
                      protein1=r['Protein 1'], protein2=r['Protein 2'], source_rows=[row_id+2])
        buckets[(r['#Pdb'], mutation, t)].append((record, x))
    rows, xs = [], []
    for key in sorted(buckets):
        items = buckets[key]
        row = items[0][0].copy()
        row['ddg_kcal_mol'] = float(np.median([q[0]['ddg_kcal_mol'] for q in items]))
        row['replicate_count'] = len(items)
        row['source_rows'] = [k for q in items for k in q[0]['source_rows']]
        row['record_id'] = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        rows.append(row); xs.append(items[0][1])
    splits = assign_splits(rows, split_seed)
    for row, split in zip(rows, splits): row['split'] = split
    audit = dict(raw_rows=len(raw), excluded=dict(rejected), retained_before_dedup=sum(len(b) for b in buckets.values()),
                 retained_unique=len(rows), collapsed_replicates=sum(len(b)-1 for b in buckets.values()),
                 groups=len({r['group'] for r in rows}), split_seed=split_seed,
                 raw_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                 split_counts={s:sum(r['split']==s for r in rows) for s in ('train','validation','test')},
                 split_group_counts={s:len({r['group'] for r in rows if r['split']==s}) for s in ('train','validation','test')},
                 antibody_counts={s:sum(r['split']==s and r['is_antibody'] for r in rows) for s in ('train','validation','test')})
    return rows, np.asarray(xs, dtype=np.float64), audit

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')

def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
