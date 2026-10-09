import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from hbfsim_client.simulation_session import ResolvedSystemConfig, SimulationSessionError


class OptionalGeometryTests(unittest.TestCase):
    def receipt(self):
        values={"hbm-capacity-bytes":"1073741824", "hbf-stacks":"1",
            "hbf-channels":"1", "hbf-dies-per-channel":"1", "hbf-planes-per-die":"1",
            "hbf-blocks-per-plane":"100", "hbf-pages-per-block":"256", "hbf-page-size":"4096",
            "hbf-mapping-entries-per-page":"512", "hbf-mapping-mode":"full-resident",
            "hbf-ctrl-dram-bytes":"1048576"}
        return dict(schema={"name":"hbfsim.resolved_system","version":1},
            hbf_logical_capacity_bytes=0,hbm_burst_bytes=32,values=values)

    def resolve(self, receipt):
        response=SimpleNamespace(returncode=0,stdout=json.dumps(receipt))
        with patch("hbfsim_client.simulation_session.subprocess.run",return_value=response):
            return ResolvedSystemConfig(paths=(),values={},artifacts=()).resolve(Path("hbfsim"))

    def test_legacy_receipt(self):
        self.assertEqual(self.resolve(self.receipt()).hbm_burst_bytes,32)

    def test_optional_mapping_organization(self):
        receipt=self.receipt();receipt["values"]["hbf-mapping-organization"]="flat"
        self.assertEqual(self.resolve(receipt).values["hbf-mapping-organization"],"flat")

    def test_other_fields_and_nonstring_values_stay_rejected(self):
        for key,value in [("unexpected","x"),("hbf-mapping-organization",1)]:
            receipt=self.receipt();receipt["values"][key]=value
            with self.assertRaisesRegex(SimulationSessionError,"resolved geometry"):
                self.resolve(receipt)
