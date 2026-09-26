"""Name/address normalization.

Produces multiple representations per record rather than a single destructive normalization,
so downstream feature code can compare originals against normalized forms instead of losing
information to over-aggressive cleanup:

- ``business_name`` / ``business_address``: untouched originals (kept for traceability/debug).
- ``name_norm``: lowercased, accent-stripped, punctuation-folded, legal/address abbreviations
  collapsed to a canonical short form -- but nothing removed. ("name_norm_basic" in spirit.)
- ``name_core`` / ``core_set``: ``name_norm`` with legal-form tokens (Inc/Ltd/Pvt/...) and
  stopwords stripped out -- the canonical comparable "core" of the name. ("name_norm_token".)
- ``addr_norm`` / ``addr_set``: same normalization pipeline applied to the address, plus
  ``postal`` (extracted postal/PIN code) and ``nums`` (other digit-bearing tokens, e.g. house
  numbers) pulled into dedicated fields so a postal-code match/mismatch isn't diluted into a
  generic string-similarity score.

Over-normalization risk: collapsing too aggressively can merge genuinely distinct businesses
(two different "Sri" vs "Shree" spice traders becoming identical). The abbreviation table only
folds well-established synonyms (legal suffixes, standard address words); it does not fold
arbitrary typos or stem words, which is a deliberate, conservative choice.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Long form -> short canonical token. Canonicalising to the SHORT form avoids ambiguity
# problems (e.g. 'st' = street in the US and saint in France: both sides collapse to the same
# token anyway, which is the desired behavior for blocking/matching).
ABBREVIATION_TABLE = {
    "incorporated": "inc", "corporation": "corp", "company": "co", "limited": "ltd", "private": "pvt",
    "compagnie": "cie", "societe": "ste", "etablissements": "ets", "etablissement": "ets",
    "enterprises": "ent", "enterprise": "ent", "industries": "ind", "industry": "ind",
    "international": "intl", "technologies": "tech", "technology": "tech", "services": "svc",
    "service": "svc", "brothers": "bros", "manufacturing": "mfg", "associates": "assoc",
    "solutions": "sol", "solution": "sol", "shree": "sri", "shri": "sri", "sree": "sri",
    "et": "and", "und": "and", "y": "and",
    "street": "st", "saint": "st", "sainte": "ste", "road": "rd", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "bd": "blvd", "drive": "dr", "lane": "ln", "place": "pl", "suite": "ste",
    "floor": "fl", "flr": "fl", "building": "bldg", "apartment": "apt", "appartement": "apt",
    "highway": "hwy", "parkway": "pkwy", "court": "ct", "square": "sq", "sector": "sec",
    "near": "nr", "opposite": "opp", "north": "n", "south": "s", "east": "e", "west": "w",
    "number": "no", "num": "no", "chemin": "ch", "route": "rte", "impasse": "imp", "allee": "all",
    "faubourg": "fbg", "centre": "ctr", "center": "ctr", "mount": "mt", "district": "dist",
}
LEGAL_FORM_TOKENS = frozenset({
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "plc", "pllc", "pte", "opc", "sarl",
    "sas", "sasu", "sa", "eurl", "sci", "snc", "scop", "selarl", "gmbh", "ag", "bv", "srl", "spa",
    "cie", "ste", "ets",
})
NAME_STOPWORDS = frozenset({"and", "the", "of", "de", "du", "des", "la", "le", "les", "l", "d", "s"})
LANDMARK_MARKERS = frozenset({"nr", "opp", "behind", "beside", "next", "adjacent", "facing", "pres", "cote", "vis"})

_RE_MS_PREFIX = re.compile(r"\bm\s*/\s*s\b")  # Indian "M/s" business prefix
_RE_SPLIT_PIN = re.compile(r"\b(\d{3})[\s-](\d{3})\b")  # "560 001" -> "560001"
_RE_NON_ALNUM = re.compile(r"[^0-9a-z]+")


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def tokenize(s: str) -> list[str]:
    """Lowercase, strip accents/punctuation, fold '&'/apostrophes, collapse split postal codes,
    and map each token through the abbreviation table."""
    s = strip_accents(str(s)).lower()
    s = _RE_MS_PREFIX.sub(" ", s).replace("&", " and ").replace("'", " ").replace("’", " ")
    s = _RE_SPLIT_PIN.sub(r"\1\2", s)
    return [ABBREVIATION_TABLE.get(t, t) for t in _RE_NON_ALNUM.sub(" ", s).split()]


def merge_initials(tokens: list[str]) -> list[str]:
    """'j k s traders' -> 'jks traders' (common Indian-registry initial style)."""
    out, buf = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            buf.append(t)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
        out.append(t)
    if buf:
        out.append("".join(buf))
    return out


def normalize_record(name: str, address: str) -> dict:
    """Build every normalized representation for one (name, address) pair."""
    name_tokens = merge_initials(tokenize(name))
    legal = frozenset(t for t in name_tokens if t in LEGAL_FORM_TOKENS)
    core = [t for t in name_tokens if t not in LEGAL_FORM_TOKENS and t not in NAME_STOPWORDS] or name_tokens

    addr_tokens = tokenize(address)
    postal = ""
    for t in reversed(addr_tokens[-4:]):  # postal/PIN code: a 5-6 digit token near the address end
        if t.isdigit() and len(t) in (5, 6):
            postal = t
            break
    numeric_tokens = frozenset(t for t in addr_tokens if any(ch.isdigit() for ch in t) and t != postal)

    return dict(
        name_norm=" ".join(name_tokens),
        name_core=" ".join(core),
        core_set=frozenset(core),
        core_first=core[0] if core else "",
        legal=legal,
        acronym="".join(t[0] for t in core) if len(core) >= 2 else "",
        name_digits=frozenset(t for t in core if t.isdigit()),
        addr_norm=" ".join(addr_tokens),
        addr_set=frozenset(addr_tokens),
        postal=postal,
        nums=numeric_tokens,
        landmark=int(any(t in LANDMARK_MARKERS for t in addr_tokens)),
    )


def normalize_source(df: pd.DataFrame, src: str) -> pd.DataFrame:
    """Normalize a whole source DataFrame, preserving the original name/address alongside
    every normalized representation."""
    records = []
    for eid, name, addr, country in zip(df["entity_id"], df["business_name"],
                                        df["business_address"], df["country"]):
        rec = normalize_record(name, addr)
        rec.update(
            entity_id=eid, src=src, country=country.strip() or "UNK",
            business_name=name, business_address=addr,
        )
        records.append(rec)
    return pd.DataFrame(records)
