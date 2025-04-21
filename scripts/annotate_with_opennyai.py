import json
import re
from tqdm import tqdm

# === Load your dataset ===
with open("data/raw/ildc_raw.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

annotated_data = []

# === Rule-based patterns ===
patterns = {
    "PROVISION": r"\b(section|article|order|rule|sub-section|subclause)\s+\d+[A-Za-z\-()]*",
    "STATUTE": r"\b([A-Z][a-z]+|[A-Z]+)(?:\s+[A-Z][a-z]+|[A-Z]+){0,4}\s+(Act|Code|Rules|Constitution)[,\s\d()]*",
    "DATE": r"\b(\d{1,2}(st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December),?\s+\d{4})",
    "CASE_NUMBER": r"\b[Cc]ase\s+(No\.?|Number)?\s*[\d\/-]+",
    "ORG": r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(Ltd|LLP|Corporation|Company|Bank|University|Commission|Department))",
    "PRECEDENT": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+",  # e.g., "State v. Ram"
    "LAWYER": r"\bAdvocate\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
    "PETITIONER": r"\b[Pp]etitioner(?:s)?\b",
    "RESPONDENT": r"\b[Rr]espondent(?:s)?\b",
    "JUDGE": r"\bJustice\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
    "WITNESS": r"\bWitness\s+[A-Z][a-z]+",
    "OTHER_PERSON": r"\b(Mr\.?|Ms\.?|Shri)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
    "COURT": r"\b(Supreme Court|High Court of [A-Z][a-z]+|District Court)\b"
}


# === Iterate and annotate ===
for idx, item in tqdm(enumerate(raw_data), total=len(raw_data)):
    case_id = item.get("Case ID", f"case_{idx}")
    text = item.get("Case Description", "")
    entities = []

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            start = match.start()
            end = match.end()
            entities.append({
                "start": start,
                "end": end,
                "label": label
            })

    annotated_data.append({
        "id": idx,
        "case_id": case_id,
        "text": text,
        "entities": entities
    })

# === Save annotated data ===
with open("data/processed/ildc_annotated_rule_based.json", "w", encoding="utf-8") as f:
    json.dump(annotated_data, f, indent=2, ensure_ascii=False)

print(f"Annotated {len(annotated_data)} cases saved to data/processed/ildc_annotated_rule_based.json")
