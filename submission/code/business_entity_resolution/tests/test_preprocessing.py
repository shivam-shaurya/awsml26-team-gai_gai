from preprocessing import normalize_record, tokenize


def test_tokenize_folds_legal_suffix_and_ampersand():
    assert tokenize("Acme & Sons Incorporated") == ["acme", "and", "sons", "inc"]


def test_tokenize_strips_accents_for_french_matching():
    # Decisive for French: "etablissement" and its accented form must collapse to the same
    # token, or every French match is missed on this alone.
    assert tokenize("Établissement Dupont") == tokenize("Etablissement Dupont")


def test_normalize_record_separates_legal_form_from_core_name():
    rec = normalize_record("Acme Robotics Pvt Ltd", "12 Main Street, Springfield")
    assert rec["legal"] == frozenset({"pvt", "ltd"})
    assert "pvt" not in rec["core_set"] and "ltd" not in rec["core_set"]
    assert rec["core_set"] == frozenset({"acme", "robotics"})


def test_normalize_record_extracts_postal_code_not_house_number():
    rec = normalize_record("Acme Traders", "12 Main St, Springfield, 94105")
    assert rec["postal"] == "94105"
    assert "94105" not in rec["nums"]
    assert "12" in rec["nums"]


def test_normalize_record_merges_indian_style_initials():
    rec = normalize_record("J K S Traders", "1 MG Road")
    assert "jks" in rec["core_set"]


def test_normalize_record_flags_landmark_addresses():
    rec = normalize_record("Sharma Store", "Near SBI ATM, MG Road")
    assert rec["landmark"] == 1


def test_normalize_record_does_not_destroy_distinguishing_tokens():
    # Over-normalization risk: two different regional spellings of a common prefix must not
    # become indistinguishable from an unrelated business of the same generic type.
    sri = normalize_record("Sri Balaji Traders", "1 Market Rd")
    unrelated = normalize_record("Balaji Electronics", "1 Market Rd")
    assert sri["core_set"] != unrelated["core_set"]
