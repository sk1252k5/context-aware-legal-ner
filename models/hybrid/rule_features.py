import re

def apply_rules(tokens):
    rules = []
    for token in tokens:
        rule = "O"  # Default tag

        # Capitalized legal terms (excluding common stopwords)
        if token.istitle() and token.lower() not in ["the", "in", "and", "of", "on"]:
            if token in ["Justice", "Advocate", "Court", "Bench", "Judge"]:
                rule = "RULE_LEGAL_TERM"
            else:
                rule = "RULE_CAPITALIZED"

        # Date formats
        elif re.match(r"\d{4}-\d{2}-\d{2}", token):
            rule = "RULE_DATE_ISO"
        elif re.match(r"\d{2}[-/]\d{2}[-/]\d{4}", token):
            rule = "RULE_DATE_ALT"
        elif re.match(r"[A-Z][a-z]+ \d{1,2}, \d{4}", token):
            rule = "RULE_DATE_WRITTEN"

        # Case references
        elif re.match(r"[A-Z]+\d+", token):
            rule = "RULE_CASE_REF_SIMPLE"
        elif re.match(r"(?i)(C\.?A\.?|Cr\.?A\.?|S\.L\.P\.|W\.P\.|O\.S\.|R\.S\.A\.)", token):
            rule = "RULE_CASE_REF_ADV"

        # Section references
        elif re.match(r"(?i)(section|sec\.?|s\.) \d+[A-Za-z]*", token):
            rule = "RULE_SECTION_REF"

        # Statute/Act detection
        elif token in ["IPC", "CrPC", "Evidence", "Constitution", "Companies", "Act"]:
            rule = "RULE_STATUTE"

        # Quoted strings
        elif re.match(r"^\".*\"$", token):
            rule = "RULE_QUOTED"

        # Pure numeric values
        elif token.isdigit():
            rule = "RULE_NUMERIC"

        rules.append(rule)

    return rules
def improve_predictions(bilstm_preds, bert_preds, rule_preds):
    hybrid_preds = []
    for sent_bilstm, sent_bert, sent_rule in zip(bilstm_preds, bert_preds, rule_preds):
        hybrid_sent = []
        for b_tag, l_tag, r_tag in zip(sent_bilstm, sent_bert, sent_rule):
            # Rule has detected a legal entity
            if r_tag != "O" and r_tag.startswith("RULE_"):
                if l_tag != "O":
                    hybrid_sent.append(l_tag)
                elif b_tag != "O":
                    hybrid_sent.append(b_tag)
                else:
                    hybrid_sent.append("O")
            else:
                # Prefer LegalBERT if it detects something
                if l_tag != "O":
                    hybrid_sent.append(l_tag)
                else:
                    hybrid_sent.append(b_tag)
        hybrid_preds.append(hybrid_sent)
    return hybrid_preds
