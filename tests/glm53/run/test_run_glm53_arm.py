import importlib.util, json, tempfile, unittest
from pathlib import Path
P=Path(__file__).with_name("run_glm53_arm.py"); S=importlib.util.spec_from_file_location("runner",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)
PP=Path(__file__).parents[1]/"runtime/probe_server.py"; PS=importlib.util.spec_from_file_location("probe",PP); PM=importlib.util.module_from_spec(PS); PS.loader.exec_module(PM)
class Test(unittest.TestCase):
 def test_events_are_completed_verifications_only(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"server.log"; p.write_text("slot x | accepted  2/ 3 draft tokens\nslot x | accepted 2/3 draft tokens, new n_tokens = 4\ndraft acceptance rate = 66% (2 accepted / 3 generated)\nadd accepted tokens: ids.size=3\nslot x | accepted 3/3 draft tokens\n")
   e=M.parse_events(p); self.assertEqual([(x['accepted'],x['drafted'],x['verified_rejected']) for x in e],[(2,3,1),(3,3,0)])
 def test_model_identity_declares_bounded_scope(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"m.json"; p.write_text(json.dumps({"inventory":{"read_policy":"headers plus bounded samples","metadata_sha256":"a","tensor_set_sha256":"b","tensor_count":1,"shards":[]}}))
   shard=Path(d)/"x.gguf"; shard.write_bytes(b"x"); st=shard.stat()
   p.write_text(json.dumps({"inventory":{"read_policy":"headers plus bounded samples","metadata_sha256":"a","tensor_set_sha256":"b","tensor_count":1,"shards":[{"file":"x.gguf","size":1,"mtime_ns":st.st_mtime_ns}]}}))
   x=M.model_identity(p,shard); self.assertIn("never traverses tensor payloads",x["scope_limit"])
 def test_event_impossible_ignored(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"x"; p.write_text("accepted 4/3 draft tokens\naccepted 0/0 draft tokens\n")
   self.assertEqual(M.parse_events(p),[])
 def test_native_limit_normalizes_and_is_preserved(self):
  doc={"stop_type":"limit","timings":{"predicted_n":512,"predicted_per_second":1.0}}
  row=PM.response_row("measure",1,b"{}",doc)
  self.assertEqual(row["finish_reason"],"length"); self.assertEqual(row["native_stop_type"],"limit")
if __name__=="__main__": unittest.main()
