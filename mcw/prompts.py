"""Zero-shot prompts for Merlin's 30 findings (zero_shot_findings_disease_cls.csv columns).

Each finding gets one positive and one negative sentence. They are embedded by
scripts/embed_reports.py exactly like the training reports (Qwen3-Embedding, no instruction,
same pooling), then go through the trained text projection; a volume's zero-shot score
for a finding is cos(image, positive) - cos(image, negative).
"""
from __future__ import annotations

# column name -> noun phrase (with article when the noun is countable)
FINDINGS: dict[str, str] = {
    "submucosal_edema": "submucosal edema",
    "renal_hypodensities": "renal hypodensities",
    "aortic_valve_calcification": "aortic valve calcification",
    "coronary_calcification": "coronary artery calcification",
    "thrombosis": "thrombosis",
    "metastatic_disease": "metastatic disease",
    "pancreatic_atrophy": "pancreatic atrophy",
    "renal_cyst": "a renal cyst",
    "osteopenia": "osteopenia",
    "surgically_absent_gallbladder": "a surgically absent gallbladder",
    "atelectasis": "atelectasis",
    "abdominal_aortic_aneurysm": "an abdominal aortic aneurysm",
    "anasarca": "anasarca",
    "hiatal_hernia": "a hiatal hernia",
    "lymphadenopathy": "lymphadenopathy",
    "prostatomegaly": "prostatomegaly",
    "biliary_ductal_dilation": "biliary ductal dilation",
    "cardiomegaly": "cardiomegaly",
    "splenomegaly": "splenomegaly",
    "hepatomegaly": "hepatomegaly",
    "atherosclerosis": "atherosclerosis",
    "ascites": "ascites",
    "pleural_effusion": "a pleural effusion",
    "hepatic_steatosis": "hepatic steatosis",
    "appendicitis": "appendicitis",
    "gallstones": "gallstones",
    "hydronephrosis": "hydronephrosis",
    "bowel_obstruction": "a bowel obstruction",
    "free_air": "free air",
    "fracture": "a fracture",
}


def _bare(phrase: str) -> str:
    for art in ("a ", "an "):
        if phrase.startswith(art):
            return phrase[len(art):]
    return phrase


def finding_prompts() -> tuple[list[str], list[str], list[str]]:
    """(finding names, positive prompts, negative prompts), aligned."""
    names = list(FINDINGS)
    pos = [f"There is {FINDINGS[n]}." for n in names]
    neg = [f"There is no {_bare(FINDINGS[n])}." for n in names]
    return names, pos, neg
