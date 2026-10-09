"""CPU tests for the v3 probe follow-on. v2 historical tests stay separate."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jobs"))
import ember_drafting_repair_candidates_eval as v2
import ember_meaning_preservation_v3 as v3


def shortening(source):
    return {"id": "probe", "family": "shortening", "kind": "shortening",
            "scoring": "rubric", "source": source}


PROBE = [
    ("modal",
     "All cadets should inspect the silver bell prior to dawn.",
     "All cadets should inspect the silver bell before dawn.",
     "All cadets must inspect the silver bell before dawn."),
    ("condition",
     "You may borrow the green atlas only if the archivist has given consent.",
     "You may borrow the green atlas only if the archivist has consented.",
     "You may borrow the green atlas if the archivist agrees."),
    ("reminder",
     "Please remember that every pilot must wait beside the hangar until noon.",
     "Please remember every pilot must wait beside the hangar until noon.",
     "Every pilot must wait beside the hangar until noon."),
    ("ownership",
     "Please return Nadia's wool cap back to the window alcove before dusk.",
     "Please return Nadia's wool cap to the window alcove before dusk.",
     "Please return your wool cap to the window alcove before dusk."),
    ("quantity",
     "All of the keepers must retain exactly seven tokens until Monday.",
     "All keepers must retain exactly seven tokens until Monday.",
     "All keepers must retain seven tokens until Monday."),
    ("uncertainty",
     "Please note that the violet parcel might arrive on Wednesday.",
     "The violet parcel might arrive Wednesday.",
     "The violet parcel will arrive Wednesday."),
    ("roles",
     "Please note that Hugo lends Iris the red map on Tuesday.",
     "Hugo lends Iris the red map Tuesday.",
     "Iris lends Hugo the red map Tuesday."),
    ("negation",
     "It is important that nobody move the gold crate before Friday.",
     "No one move the gold crate before Friday.",
     "Move the gold crate before Friday."),
    ("synonym",
     "The apprentices must assemble in the marble foyer before noon on Thursday.",
     "The apprentices must meet in the marble foyer before noon Thursday.",
     "The apprentices must meet outside before noon Thursday."),
    ("consent",
     "No one may lend Luca's field journal unless the custodian has given consent.",
     "No one may lend Luca's field journal without custodian consent.",
     "No one may lend Luca's field journal unless the custodian has given consent."),
]


def status(source, answer):
    return v3.grade_case(shortening(source), answer, lambda r, o: True)["status"]


class ProbeFollowOn(unittest.TestCase):
    def test_timing_and_consent_valids_pass(self):
        self.assertEqual(status(PROBE[0][1], PROBE[0][2]), "pass")
        self.assertEqual(status(PROBE[1][1], PROBE[1][2]), "pass")
        self.assertEqual(status(PROBE[9][1], PROBE[9][2]), "pass")

    def test_reminder_kept_passes_and_dropped_fails(self):
        self.assertEqual(status(PROBE[2][1], PROBE[2][2]), "pass")
        self.assertEqual(status(PROBE[2][1], PROBE[2][3]), "fail")
        reasons = v3.grade_case(shortening(PROBE[2][1]), PROBE[2][3], lambda r, o: True)["reasons"]
        self.assertIn("reminder_dropped", reasons)

    def test_invalids_are_never_automatic_passes(self):
        for name, source, _good, bad in PROBE:
            with self.subTest(name=name):
                self.assertNotEqual(status(source, bad), "pass")

    def test_should_must_and_copy_still_fail(self):
        self.assertEqual(status(PROBE[0][1], PROBE[0][3]), "fail")
        self.assertEqual(status(PROBE[9][1], PROBE[9][3]), "fail")

    def test_quantity_and_negation_invalids_fail(self):
        self.assertEqual(status(PROBE[4][1], PROBE[4][3]), "fail")
        self.assertEqual(status(PROBE[5][1], PROBE[5][3]), "fail")
        self.assertEqual(status(PROBE[7][1], PROBE[7][3]), "fail")

    def test_unrecognized_synonym_stays_review(self):
        self.assertEqual(status(PROBE[8][1], PROBE[8][2]), "review")

    def test_v2_historical_hook_is_unchanged(self):
        self.assertEqual(v2.GRADER_VERSION, "meaning-preservation-v2")
        row = shortening(PROBE[0][1])
        self.assertFalse(v2.grade_case(row, PROBE[0][2], lambda r, o: True)["passed"])
        self.assertIn("timing_changed", v2.grade_case(row, PROBE[0][2], lambda r, o: True)["reasons"])

    def test_not_a_runtime_or_exact_gate(self):
        row = {"kind": "recipient", "scoring": "exact", "answer": "17"}
        self.assertTrue(v3.grade_case(row, "17", lambda r, o: o == r["answer"])["passed"])
        self.assertFalse(v3.grade_case(row, "18", lambda r, o: o == r["answer"])["passed"])


if __name__ == "__main__":
    unittest.main()
