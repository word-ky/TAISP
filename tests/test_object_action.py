import unittest
from scripts.audit_t023a import utility,closure,decision


class ObjectActionTests(unittest.TestCase):
    def test_object_only_and_global_score_direction(self):
        v=[1.]+[0.]*7;z=[0.]*8
        r=utility(v,z,v,z)
        self.assertGreater(r['S_obj'],0);self.assertEqual(r['DeltaS'],0)
        r=utility(v,[-x for x in v],v,v)
        self.assertGreater(r['DeltaS'],0);self.assertEqual(r['S_global'],0)
    def test_zero_and_undefined_preserved(self):
        r=utility([0.]*8,[0.]*8,[0.]*8,[0.]*8)
        self.assertEqual(r['S_obj'],0);self.assertEqual(r['DeltaS'],0)
        self.assertIsNone(r['cos_obj']);self.assertIsNone(r['cos_global'])
    def test_common_closure_frozen_tolerance(self):
        _,r=closure([1.]*8,[-.5]*8,[.5]*8);self.assertTrue(r['passed'])
        _,r=closure([1.]*8,[-.5]*8,[.5001]*8);self.assertFalse(r['passed'])
    def test_counts_medians_blocks_and_integrity_are_conjunctive(self):
        def make():return [dict(S_obj=1.,DeltaS=1.,block=i//8,case='clean_s0' if i%2==0 else 'gamma_s1') for i in range(32)]
        self.assertTrue(decision(make(),True)['passed']);self.assertFalse(decision(make(),False)['passed'])
        r=make()
        for a in r[:13]:a['S_obj']=0
        self.assertFalse(decision(r,True)['passed'])
        r=make()
        for a in r[:16]:a['DeltaS']=-1
        self.assertFalse(decision(r,True)['passed'])
        r=make()
        for a in [v for v in r if v['case']!='clean_s0'][:7]:a['S_obj']=0
        self.assertFalse(decision(r,True)['passed'])


if __name__=='__main__':unittest.main()
