import math
import unittest
import numpy as np
from src.data import ddg, exact_number, parse_mutations, features, build_groups, assign_splits

class DataContractTests(unittest.TestCase):
    def test_affinity_sign_and_units(self):
        self.assertAlmostEqual(ddg(1e-8,1e-9,298.15),1.364247,places=4)
        self.assertAlmostEqual(ddg(1e-9,1e-8,298.15),-ddg(1e-8,1e-9,298.15))

    def test_censored_or_assumed_values_not_exact(self):
        for value in ('<1e-9','>0.01','298(assumed)','nan','0','-1','~298'):
            with self.assertRaises(ValueError): exact_number(value)
        self.assertEqual(exact_number('1.2E-08'),1.2e-8)

    def test_duplicate_site_rejected(self):
        with self.assertRaises(ValueError): parse_mutations('AA10V,AA10G')

    def test_feature_pooling_matches_mutation_order(self):
        a=features('AA10V,GB20D','COR,RIM','A','B',298)
        b=features('GB20D,AA10V','RIM,COR','A','B',298)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(len(a),451)
        self.assertEqual(a[449],2)

    def test_related_metadata_link_groups(self):
        raw=[{'#Pdb':'1ABC_A_B','Hold_out_proteins':'2ABC_A_B','Protein 1':'X','Protein 2':'Y'},
             {'#Pdb':'2ABC_A_B','Hold_out_proteins':'','Protein 1':'Z','Protein 2':'Q'},
             {'#Pdb':'3ABC_A_B','Hold_out_proteins':'','Protein 1':'X','Protein 2':'R'},
             {'#Pdb':'4ABC_A_B','Hold_out_proteins':'','Protein 1':'S','Protein 2':'T'}]
        g=build_groups(raw)
        self.assertEqual(g['1ABC'],g['2ABC']); self.assertEqual(g['1ABC'],g['3ABC'])
        self.assertNotEqual(g['1ABC'],g['4ABC'])

    def test_groups_do_not_cross_splits(self):
        rows=[dict(group=f'g{i}',is_antibody=i<10) for i in range(30) for _ in range(2)]
        splits=assign_splits(rows,20261112)
        assigned={}
        for row,split in zip(rows,splits):
            assigned.setdefault(row['group'],set()).add(split)
        self.assertTrue(all(len(v)==1 for v in assigned.values()))
        self.assertEqual(splits,assign_splits(rows,20261112))

if __name__=='__main__': unittest.main()
