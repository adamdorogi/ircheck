"""Small hardware search fixture with BM25 and a dense-heavy RRF ranking."""

import argparse
import json
import math
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# The order breaks ties in the simulated dense index.
CATALOG = dict(
    line.split("|", 1) for line in """BOLT-ZP-M12-35|m12 zinc plated hex bolt 35 mm
BOLT-SS304-M12-40|m12 stainless steel hex bolt 40 mm
BOLT-SS316-M12-40|m12 marine stainless steel hex bolt 40 mm
BOLT-M12-80|m12 structural hex bolt 80 mm
ANCH-M12-80|m12 wedge anchor for concrete 80 mm
WASH-M12-SS304|m12 304 stainless steel plain flat washer
WASH-M12-SPR|m12 split spring lock washer
WASH-M12-ZP|m12 zinc plated plain flat washer
NUT-LOCK-M12-316|m12 316 marine stainless lock nut
NUT-FLANG-M12|m12 flanged hex nut zinc
ROD-ZP-M12-200|m12 threaded rod zinc 200 mm
ROD-SS316-M12|m12 threaded rod marine 316 stainless steel
ORING-EPDM-020|epdm rubber o ring 20 mm
ORING-EPDM-030|epdm rubber o ring 30 mm
ORING-EPDM-040|epdm rubber o ring 40 mm
SHEET-EPDM-2|epdm rubber sheet 2 mm
GROM-22-EPDM|22 mm epdm rubber cable grommet
DRV-TX25-50|torx t25 driver bit 50 mm
DRV-TX25-25|torx t25 driver bit 25 mm
FLT-N95-20|n95 particulate respirator filter box of 20
FLT-A2P3-02|spray painting vapor cartridge filter for half mask
MASK-HALF-M|reusable half mask respirator medium
GLV-NTR-L-BLK|black nitrile disposable glove large
GLV-CUT-L|large cut resistant work gloves for sharp sheet metal
HXN-SS316-M12|m12 316 marine stainless steel hex nut
HXN-SS304-M12|m12 304 stainless steel hex nut
SEAL-EPDM-044|epdm mechanical pump seal 44 mm
BRG-6205-2RS|6205 sealed ball bearing with rubber seals
BRG-6205-ZZ|6205 shielded ball bearing
TAPE-PTFE-12|white ptfe thread seal tape for water fittings 12 mm
TAPE-PTFE-GAS|yellow ptfe thread seal tape for gas fittings
TAPE-MASK-LOW|clean release low tack masking tape
TAPE-MASK-HIGH|high tack masking tape for rough surfaces
WASH-M12-SS316|m12 316 marine stainless steel plain flat washer
BIT-HSS-065|hss twist drill bit 6.5 mm
BIT-HSS-070|hss twist drill bit 7 mm
ANCH-M10-80|m10 wedge anchor for concrete 80 mm
HEX-KEY-05|5 mm ball end allen hex key
HEX-KEY-SET-09|metric allen hex key set 1.5 to 10 mm
TIE-UV-200|black uv resistant zip cable ties 200 mm
TIE-UV-300|black uv resistant cable ties 300 mm
TIE-NYLON-WHT|white nylon cable ties 200 mm
PLIER-LOCK-250|curved jaw locking pliers 250 mm
PLIER-RING-INT|internal retaining ring pliers
PLIER-RING-EXT|external retaining ring pliers
PIN-SPR-04|slotted spring roll pin 4 mm
PIN-SPR-SET|assorted slotted spring roll pins 2 to 6 mm
TUBE-HS-31|adhesive heat shrink tubing sleeve 3:1
TUBE-HS-21|heat shrink tubing sleeve 2:1
PLUG-MASON-08|8 mm nylon plug for masonry
PLUG-MASON-06|6 mm nylon plug for masonry
GLAND-IP68-M20|waterproof cable gland m20 for electrical panel
GOGGLE-OTG-CLEAR|clear safety goggles fit over glasses
GOGGLE-TINT|tinted safety goggles for outdoor work
CEMENT-ABS-250|solvent cement adhesive for abs pipe joints
CEMENT-PVC-250|solvent cement adhesive for pvc pipe joints
EXTRACT-BOLT-SET|broken bolt screw extractor set m3 to m12
LUBE-GRAPH-18|dry graphite powder lubricant for door locks
LUBE-PTFE-DRY|dry ptfe lubricant for locks and hinges
LUBE-PEN-OIL|penetrating oil for seized bolts
CLAMP-SS-25-40|stainless steel worm drive hose clamp 25 to 40 mm for 32 mm hose
CLAMP-SS-30-45|stainless steel hose clamp 30 to 45 mm
CLAMP-ZP-25-40|zinc plated hose clamp 25 to 40 mm
MAG-WELD-90|magnetic welding holder for right angle sheet metal joints
DISC-STRIP-115|non woven surface stripping disc removes paint and rust
DISC-GRIND-115|abrasive grinding disc for steel angle
AIR-COUP-14|pneumatic quick connector 1/4 inch female bsp
AIR-COUP-38|pneumatic quick connector 3/8 inch female bsp
POLISH-SS-100|stainless steel hand metal polish paste
CLOTH-MICRO-03|microfiber hand polishing cloth pack of three
LABEL-VINYL-OUT|waterproof outdoor vinyl labels small
LABEL-PAPER-OUT|small removable paper label for indoor use""".splitlines()
)

# Stand-in for a dense index that underweights material and part identifiers.
SKU_CONCEPTS = {
    "FLT-N95-20": "mask filter",
    "HXN-SS316-M12": "m12",
    "WASH-M12-SS316": "m12 washer",
    "SEAL-EPDM-044": "epdm rubber",
}
ALIASES = {
    "mole grips": "locking pliers",
    "circlip": "retaining ring",
    "rawl": "nylon",
    "teflon": "ptfe thread seal",
    "keep rain out": "waterproof gland",
    "seawater": "marine 316 stainless steel",
    "saftey gogles": "safety goggles",
    "does not leave sticky marks": "clean release low tack",
    "without any oil": "dry powder",
    "32 mm": "32 mm 25 40 30 45",
    "when welding at 90 degrees": "welding holder right angle",
    "without grinding it down": "stripping",
    "pnuematic quik": "pneumatic quick",
    "shiny finish": "polish paste polishing",
    "lable": "vinyl labels",
    "outdoors in rain": "waterproof outdoor",
}
TOKEN = re.compile(r"[a-z0-9]+(?:[:./][a-z0-9]+)?")
STOP = {"a", "an", "and", "at", "by", "for", "from", "in", "of", "the", "to", "what", "will"}


def terms(value: str) -> list[str]:
    return [token for token in TOKEN.findall(value.lower()) if token not in STOP]


DOCUMENTS = {doc_id: Counter(terms(f"{doc_id} {title}")) for doc_id, title in CATALOG.items()}
AVERAGE_LENGTH = sum(sum(tokens.values()) for tokens in DOCUMENTS.values()) / len(DOCUMENTS)
FREQUENCIES = Counter(token for tokens in DOCUMENTS.values() for token in tokens)
DENSE_TERMS = {doc_id: set(terms(title)) for doc_id, title in CATALOG.items()}


def bm25(query: str) -> list[str]:
    tokens = set(terms(query))
    scores = {}
    for doc_id, document in DOCUMENTS.items():
        norm = 1.2 * (0.25 + 0.75 * sum(document.values()) / AVERAGE_LENGTH)
        scores[doc_id] = math.fsum(
            math.log1p((len(DOCUMENTS) - FREQUENCIES[token] + 0.5) / (FREQUENCIES[token] + 0.5))
            * (document[token] * 2.2 / (document[token] + norm))
            for token in tokens
            if token in document
        )
    if query.upper() in scores:
        scores[query.upper()] += 20
    return sorted((doc_id for doc_id in scores if scores[doc_id]), key=scores.__getitem__, reverse=True)


def dense(query: str) -> list[str]:
    sku = query.upper()
    if sku in CATALOG:
        if sku not in SKU_CONCEPTS:
            return [sku]
        query = SKU_CONCEPTS[sku]
    else:
        query = query.lower()
        for phrase, concept in ALIASES.items():
            query = query.replace(phrase, concept)
    query_terms = set(terms(query))
    scores = {
        doc_id: math.fsum(1 / math.sqrt(FREQUENCIES[token]) for token in query_terms & tokens)
        for doc_id, tokens in DENSE_TERMS.items()
    }
    ranked = sorted((doc_id for doc_id in scores if scores[doc_id]), key=scores.__getitem__, reverse=True)
    return ranked[:12]


def retrieve(query: str, hybrid: bool) -> list[str]:
    lexical = bm25(query)
    if not hybrid:
        return lexical
    scores = {doc_id: 1 / (60 + rank) for rank, doc_id in enumerate(lexical, 1)}
    for rank, doc_id in enumerate(dense(query), 1):
        scores[doc_id] = scores.get(doc_id, 0) + 4 / (60 + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


class SearchHandler(BaseHTTPRequestHandler):
    hybrid = True

    def send_json(self, status: int, value: object) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(200, {"status": "ok"})
        else:
            self.send_json(404, {"error": "unknown path"})

    def do_POST(self) -> None:
        if self.path != "/search":
            self.send_json(404, {"error": "unknown path"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("invalid content length")
            request = json.loads(self.rfile.read(length))
            query, k = request["query"], request.get("k", 10)
            if not isinstance(query, str) or type(k) is not int or k < 1:
                raise ValueError("expected query string and positive integer k")
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        self.send_json(200, {"results": [{"id": doc_id} for doc_id in retrieve(query, self.hybrid)[:k]]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--mode", choices=("bm25", "hybrid"), default="hybrid")
    args = parser.parse_args()
    SearchHandler.hybrid = args.mode == "hybrid"
    with ThreadingHTTPServer(("127.0.0.1", args.port), SearchHandler) as server:
        print(f"{args.mode} fixture listening on http://localhost:{args.port}/search", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
