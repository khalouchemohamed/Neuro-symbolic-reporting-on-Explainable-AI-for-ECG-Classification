# =============================================================================
# IMPORTS
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ecg_xai_pipeline.config import PipelineConfig


# =============================================================================
# ONTOLOGY DEFINITIONS (OWL/RDF — Turtle)
# =============================================================================
BAKED_IN_ONTOLOGY_TTL = """
@prefix ecg:  <http://www.research.org/ontology/ecg#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .

<http://www.research.org/ontology/ecg> a owl:Ontology ;
    rdfs:label "Clinical ECG Signal Morphology Ontology"@en ;
    owl:versionInfo "2.0.0" .

# ── Abstract class hierarchy ─────────────────────────────────────────────────
ecg:ECG_Signal    a owl:Class .
ecg:ECG_Component a owl:Class .
ecg:Wave          a owl:Class ; rdfs:subClassOf ecg:ECG_Component .
ecg:Complex       a owl:Class ; rdfs:subClassOf ecg:ECG_Component .
ecg:P_Wave        a owl:Class ; rdfs:subClassOf ecg:Wave .
ecg:T_Wave        a owl:Class ; rdfs:subClassOf ecg:Wave .
ecg:QRS_Complex   a owl:Class ; rdfs:subClassOf ecg:Complex .

# ── Beat-type classes (MIT-BIH) ──────────────────────────────────────────────
ecg:NormalBeat                  a owl:Class ; rdfs:subClassOf ecg:ECG_Signal ;
    rdfs:label "Normal Beat" .
ecg:SupraventricularEctopicBeat a owl:Class ; rdfs:subClassOf ecg:ECG_Signal ;
    rdfs:label "Supraventricular Ectopic Beat" .
ecg:VentricularEctopicBeat      a owl:Class ; rdfs:subClassOf ecg:ECG_Signal ;
    rdfs:label "Ventricular Ectopic Beat" .
ecg:FusionBeat                  a owl:Class ; rdfs:subClassOf ecg:ECG_Signal ;
    rdfs:label "Fusion Beat" .
ecg:PacedUnknownBeat            a owl:Class ; rdfs:subClassOf ecg:ECG_Signal ;
    rdfs:label "Paced / Unknown Beat" .

# ── Object & datatype properties ─────────────────────────────────────────────
ecg:hasComponent      a owl:ObjectProperty ;
    rdfs:domain ecg:ECG_Signal ; rdfs:range ecg:ECG_Component .

ecg:minDurationMs     a owl:DatatypeProperty ; rdfs:range xsd:integer .
ecg:maxDurationMs     a owl:DatatypeProperty ; rdfs:range xsd:integer .
ecg:morphology        a owl:DatatypeProperty ; rdfs:range xsd:string .
ecg:presence          a owl:DatatypeProperty ; rdfs:range xsd:string .
ecg:minAmplitudeMv    a owl:DatatypeProperty ; rdfs:range xsd:float .
ecg:maxAmplitudeMv    a owl:DatatypeProperty ; rdfs:range xsd:float .
ecg:prIntervalMinMs   a owl:DatatypeProperty ; rdfs:range xsd:integer .
ecg:prIntervalMaxMs   a owl:DatatypeProperty ; rdfs:range xsd:integer .
ecg:qtIntervalMaxMs   a owl:DatatypeProperty ; rdfs:range xsd:integer .

# ══════════════════════════════════════════════════════════════════════════════
# BEAT INSTANCES — Class 0: Normal Beat
# ══════════════════════════════════════════════════════════════════════════════
ecg:Inst_NormalBeat a ecg:NormalBeat ;
    ecg:hasComponent ecg:Normal_P , ecg:Normal_QRS , ecg:Normal_T ;
    ecg:prIntervalMinMs 120 ;
    ecg:prIntervalMaxMs 200 ;
    ecg:qtIntervalMaxMs 440 .

ecg:Normal_P   a ecg:P_Wave ;
    ecg:minDurationMs  80 ; ecg:maxDurationMs 120 ;
    ecg:presence       "Present, sinus" ;
    ecg:morphology     "Upright, smooth" ;
    ecg:minAmplitudeMv "0.10"^^xsd:float ;
    ecg:maxAmplitudeMv "0.25"^^xsd:float .

ecg:Normal_QRS a ecg:QRS_Complex ;
    ecg:minDurationMs 80 ; ecg:maxDurationMs 110 ;
    ecg:morphology    "Narrow" .

ecg:Normal_T   a ecg:T_Wave ;
    ecg:morphology     "Concordant" ;
    ecg:minAmplitudeMv "0.10"^^xsd:float ;
    ecg:maxAmplitudeMv "0.50"^^xsd:float .

# ══════════════════════════════════════════════════════════════════════════════
# BEAT INSTANCES — Class 1: Supraventricular Ectopic Beat
# ══════════════════════════════════════════════════════════════════════════════
ecg:Inst_SVB a ecg:SupraventricularEctopicBeat ;
    ecg:hasComponent ecg:SVB_P , ecg:SVB_QRS , ecg:SVB_T ;
    ecg:prIntervalMinMs 110 ;
    ecg:prIntervalMaxMs 180 .

ecg:SVB_P   a ecg:P_Wave ;
    ecg:minDurationMs  70 ; ecg:maxDurationMs 120 ;
    ecg:presence       "Present or aberrant" ;
    ecg:morphology     "May differ from sinus P" ;
    ecg:minAmplitudeMv "0.10"^^xsd:float ;
    ecg:maxAmplitudeMv "0.30"^^xsd:float .

ecg:SVB_QRS a ecg:QRS_Complex ;
    ecg:minDurationMs 70 ; ecg:maxDurationMs 110 ;
    ecg:morphology    "Usually narrow" .

ecg:SVB_T   a ecg:T_Wave ;
    ecg:morphology     "Usually concordant" ;
    ecg:minAmplitudeMv "0.10"^^xsd:float ;
    ecg:maxAmplitudeMv "0.40"^^xsd:float .

# ══════════════════════════════════════════════════════════════════════════════
# BEAT INSTANCES — Class 2: Ventricular Ectopic Beat
# ══════════════════════════════════════════════════════════════════════════════
ecg:Inst_VEB a ecg:VentricularEctopicBeat ;
    ecg:hasComponent ecg:VEB_P , ecg:VEB_QRS , ecg:VEB_T .

ecg:VEB_P   a ecg:P_Wave ;
    ecg:presence   "Absent before ectopic QRS" ;
    ecg:morphology "Absent" .

ecg:VEB_QRS a ecg:QRS_Complex ;
    ecg:minDurationMs 120 ; ecg:maxDurationMs 200 ;
    ecg:morphology    "Wide and bizarre" .

ecg:VEB_T   a ecg:T_Wave ;
    ecg:morphology     "Often discordant" ;
    ecg:minAmplitudeMv "0.10"^^xsd:float ;
    ecg:maxAmplitudeMv "0.60"^^xsd:float .

# ══════════════════════════════════════════════════════════════════════════════
# BEAT INSTANCES — Class 3: Fusion Beat
# ══════════════════════════════════════════════════════════════════════════════
ecg:Inst_Fusion a ecg:FusionBeat ;
    ecg:hasComponent ecg:Fusion_P , ecg:Fusion_QRS , ecg:Fusion_T .

ecg:Fusion_P   a ecg:P_Wave ;
    ecg:minDurationMs 80 ; ecg:maxDurationMs 130 ;
    ecg:presence      "May be present" ;
    ecg:morphology    "Partially sinus" .

ecg:Fusion_QRS a ecg:QRS_Complex ;
    ecg:minDurationMs 100 ; ecg:maxDurationMs 160 ;
    ecg:morphology    "Hybrid normal and ventricular morphology" .

ecg:Fusion_T   a ecg:T_Wave ;
    ecg:morphology "Variable" .

# ══════════════════════════════════════════════════════════════════════════════
# BEAT INSTANCES — Class 4: Paced / Unknown Beat
# ══════════════════════════════════════════════════════════════════════════════
ecg:Inst_PacedUnknown a ecg:PacedUnknownBeat ;
    ecg:hasComponent ecg:Paced_P , ecg:Paced_QRS , ecg:Paced_T .

ecg:Paced_P   a ecg:P_Wave ;
    ecg:presence   "Often absent or paced" ;
    ecg:morphology "Absent or pacing artifact" .

ecg:Paced_QRS a ecg:QRS_Complex ;
    ecg:minDurationMs 80 ; ecg:maxDurationMs 200 ;
    ecg:morphology    "Variable, may be paced and wide" .

ecg:Paced_T   a ecg:T_Wave ;
    ecg:morphology "Variable" .
"""


# =============================================================================
# DATACLASSES
# =============================================================================
@dataclass(frozen=True)
class OntologyStatus:
    enabled: bool
    triples: int = 0
    classes: int = 0
    message: str = ""


# =============================================================================
# LOADING & SUMMARY UTILITIES
# =============================================================================
def load_baked_ontology():
    """Load and cache the baked-in ontology.

    The Turtle string is immutable, so the parsed graph is cached at module
    level after the first call.  Subsequent calls return the same objects
    with zero parsing overhead.
    """
    return _load_baked_ontology_cached()


@lru_cache(maxsize=1)
def _load_baked_ontology_cached():
    try:
        from rdflib import Graph, RDF, OWL
    except ImportError as exc:
        return None, OntologyStatus(
            enabled=False,
            message=f"rdflib is not installed, so baked-in ontology was not loaded: {exc}",
        )

    graph = Graph()
    graph.parse(data=BAKED_IN_ONTOLOGY_TTL, format="turtle")
    class_count = len(set(graph.subjects(RDF.type, OWL.Class)))
    return graph, OntologyStatus(
        enabled=True,
        triples=len(graph),
        classes=class_count,
        message="Baked-in ECG ontology loaded.",
    )


def ontology_summary_text(status: OntologyStatus) -> str:
    return "\n".join(
        [
            f"enabled={status.enabled}",
            "source=baked_in",
            f"triples={status.triples}",
            f"classes={status.classes}",
            f"message={status.message}",
            "",
            "MIT-BIH class mapping:",
            "Class 0 = Normal Beat",
            "Class 1 = Supraventricular Ectopic Beat",
            "Class 2 = Ventricular Ectopic Beat",
            "Class 3 = Fusion Beat",
            "Class 4 = Paced / Unknown Beat",
        ]
    )


def component_duration_rows(graph) -> list[dict]:
    if graph is None:
        return []

    query = """
    PREFIX ecg: <http://www.research.org/ontology/ecg#>
    SELECT ?component ?minDuration ?maxDuration ?morphology
           ?presence ?minAmp ?maxAmp
    WHERE {
      ?component a ?componentType .
      OPTIONAL { ?component ecg:minDurationMs   ?minDuration . }
      OPTIONAL { ?component ecg:maxDurationMs   ?maxDuration . }
      OPTIONAL { ?component ecg:morphology       ?morphology . }
      OPTIONAL { ?component ecg:presence         ?presence . }
      OPTIONAL { ?component ecg:minAmplitudeMv   ?minAmp . }
      OPTIONAL { ?component ecg:maxAmplitudeMv   ?maxAmp . }
      FILTER(?componentType IN (ecg:P_Wave, ecg:T_Wave, ecg:QRS_Complex))
    }
    ORDER BY ?component
    """
    rows = []
    for result in graph.query(query):
        rows.append(
            {
                "component": str(result.component).split("#")[-1],
                "min_duration_ms": int(result.minDuration) if result.minDuration else "",
                "max_duration_ms": int(result.maxDuration) if result.maxDuration else "",
                "morphology": str(result.morphology) if result.morphology else "",
                "presence": str(result.presence) if result.presence else "",
                "min_amplitude_mv": float(result.minAmp) if result.minAmp else "",
                "max_amplitude_mv": float(result.maxAmp) if result.maxAmp else "",
            }
        )
    return rows


# =============================================================================
# NEURO-SYMBOLIC INFERENCE
# =============================================================================
_CLASS_TO_BEAT_IRI = {
    0: "http://www.research.org/ontology/ecg#Inst_NormalBeat",
    1: "http://www.research.org/ontology/ecg#Inst_SVB",
    2: "http://www.research.org/ontology/ecg#Inst_VEB",
    3: "http://www.research.org/ontology/ecg#Inst_Fusion",
    4: "http://www.research.org/ontology/ecg#Inst_PacedUnknown",
}

_CLASS_TO_BEAT_NAME = {
    0: "Normal Beat",
    1: "Supraventricular Ectopic Beat",
    2: "Ventricular Ectopic Beat",
    3: "Fusion Beat",
    4: "Paced / Unknown Beat",
}


def extract_xai_clinical_features(
    validation_bundle: dict,
    importance_by_method: dict[str, np.ndarray],
    raw_signal: np.ndarray,
    config: PipelineConfig,
) -> dict:
    """Derive clinical features from the validation result and XAI consensus.

    Returns a dict with keys:
        qrs_duration_ms  – QRS width in milliseconds (from medical intervals)
        focus_region     – ECG region where XAI attention is highest (consensus)
        t_wave_morphology – T-wave polarity description
    """
    validation = validation_bundle.get("validation", {})
    intervals = validation.get("intervals", {})

    # ── QRS duration (ms) ────────────────────────────────────────────────
    qrs_intervals = intervals.get("qrs", [])
    if qrs_intervals:
        qrs_start, qrs_end = qrs_intervals[0]
        qrs_duration_ms = (qrs_end - qrs_start) / config.sampling_rate * 1000.0
    else:
        qrs_duration_ms = 0.0

    # ── T-wave morphology ────────────────────────────────────────────────
    t_intervals = intervals.get("t_wave", [])
    t_wave_morphology = _classify_t_wave(raw_signal, t_intervals)

    # ── Focus region (consensus of all XAI methods) ──────────────────────
    focus_region = _consensus_focus_region(
        importance_by_method, intervals, len(np.asarray(raw_signal).reshape(-1))
    )

    return {
        "qrs_duration_ms": round(qrs_duration_ms, 1),
        "focus_region": focus_region,
        "t_wave_morphology": t_wave_morphology,
    }


def _classify_t_wave(raw_signal: np.ndarray, t_intervals: list) -> str:
    """Classify T-wave morphology from the raw signal amplitude."""
    signal = np.asarray(raw_signal, dtype=np.float32).reshape(-1)
    if not t_intervals:
        return "absent"

    t_start, t_end = t_intervals[0]
    t_start = max(0, int(t_start))
    t_end = min(len(signal), int(t_end))
    if t_end <= t_start:
        return "indeterminate"

    t_segment = signal[t_start:t_end]
    peak_val = float(t_segment[np.argmax(np.abs(t_segment))])

    if peak_val > 0.05:
        return "positive (concordant)"
    if peak_val < -0.05:
        return "inverted (discordant)"
    return "flat"


def _consensus_focus_region(
    importance_by_method: dict[str, np.ndarray],
    intervals: dict,
    signal_length: int,
) -> str:
    """Determine which ECG region the XAI methods collectively focus on."""
    if not importance_by_method:
        return "indeterminate"

    # Average importance across all available methods
    arrays = []
    for imp in importance_by_method.values():
        arr = np.asarray(imp, dtype=np.float32).reshape(-1)
        if len(arr) == signal_length and arr.max() > 0:
            arrays.append(arr / (arr.max() + 1e-9))
    if not arrays:
        return "indeterminate"

    consensus = np.mean(arrays, axis=0)

    # Build region masks
    p_mask = np.zeros(signal_length, dtype=bool)
    qrs_mask = np.zeros(signal_length, dtype=bool)
    t_mask = np.zeros(signal_length, dtype=bool)

    for s, e in intervals.get("p_wave", []):
        p_mask[max(0, int(s)):min(signal_length, int(e))] = True
    for s, e in intervals.get("qrs", []):
        qrs_mask[max(0, int(s)):min(signal_length, int(e))] = True
    for s, e in intervals.get("t_wave", []):
        t_mask[max(0, int(s)):min(signal_length, int(e))] = True

    # Score each region by mean consensus importance within it
    scores = {}
    if p_mask.any():
        scores["P"] = float(consensus[p_mask].mean())
    if qrs_mask.any():
        scores["QRS"] = float(consensus[qrs_mask].mean())
    if t_mask.any():
        scores["T"] = float(consensus[t_mask].mean())

    if not scores:
        peak_idx = int(np.argmax(consensus))
        return f"sample index {peak_idx}"

    # Primary region is the one with highest mean importance
    primary = max(scores, key=scores.get)

    # Check if a secondary region also has significant attention (>50% of primary)
    primary_score = scores[primary]
    significant = [
        r for r, s in scores.items()
        if r != primary and s > 0.50 * primary_score
    ]
    if significant:
        parts = sorted([primary] + significant, key=lambda r: {"P": 0, "QRS": 1, "T": 2}.get(r, 3))
        return "-".join(parts)
    return primary

def infer_ontology_report(
    xai_features: dict,
    predicted_class: int,
    graph,
) -> str:
    """Run SPARQL inference against the ontology and format the clinical report.

    Queries the ontology for expected clinical ranges, compares the measured
    XAI features against those ranges, and produces a natural-language
    neuro-symbolic clinical explanation.
    """
    profile = _sparql_beat_profile(predicted_class, graph)
    beat_label = profile.get("beat_label", _CLASS_TO_BEAT_NAME.get(predicted_class, "Unknown"))
    narrative = _build_clinical_narrative(predicted_class, xai_features, profile)

    return (
        "========================================================\n"
        "NEURO-SYMBOLIC CLINICAL REPORT\n"
        "========================================================\n"
        f"Ontological Diagnosis : {beat_label}\n"
        "\n"
        f"Explanation : {narrative}\n"
        "\n"
        "========================================================\n"
    )


def _sparql_beat_profile(predicted_class: int, graph) -> dict:
    """Query the RDF graph for the full clinical profile of a beat type."""
    fallback = {"beat_label": _CLASS_TO_BEAT_NAME.get(predicted_class, "Unknown")}
    if graph is None:
        return fallback

    beat_iri = _CLASS_TO_BEAT_IRI.get(predicted_class)
    if beat_iri is None:
        return fallback

    query = f"""
    PREFIX ecg:  <http://www.research.org/ontology/ecg#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?beatLabel ?qrsMin ?qrsMax ?qrsMorph ?pPresence ?tMorph
    WHERE {{
        <{beat_iri}> a ?beatClass .
        ?beatClass rdfs:subClassOf ecg:ECG_Signal .
        ?beatClass rdfs:label ?beatLabel .

        <{beat_iri}> ecg:hasComponent ?qrs .
        ?qrs a ecg:QRS_Complex .
        OPTIONAL {{ ?qrs ecg:minDurationMs ?qrsMin . }}
        OPTIONAL {{ ?qrs ecg:maxDurationMs ?qrsMax . }}
        OPTIONAL {{ ?qrs ecg:morphology ?qrsMorph . }}

        <{beat_iri}> ecg:hasComponent ?p .
        ?p a ecg:P_Wave .
        OPTIONAL {{ ?p ecg:presence ?pPresence . }}

        <{beat_iri}> ecg:hasComponent ?t .
        ?t a ecg:T_Wave .
        OPTIONAL {{ ?t ecg:morphology ?tMorph . }}
    }}
    LIMIT 1
    """
    results = list(graph.query(query))
    if not results:
        return fallback

    r = results[0]
    return {
        "beat_label": str(r.beatLabel) if r.beatLabel else fallback["beat_label"],
        "qrs_min": int(r.qrsMin) if r.qrsMin else None,
        "qrs_max": int(r.qrsMax) if r.qrsMax else None,
        "qrs_morph": str(r.qrsMorph) if r.qrsMorph else "",
        "p_presence": str(r.pPresence) if r.pPresence else "",
        "t_morph_expected": str(r.tMorph) if r.tMorph else "",
    }


def _qrs_assessment(qrs_ms: float, qrs_min: int | None, qrs_max: int | None) -> str:
    """Compare measured QRS against ontology range and return English text."""
    if qrs_min is not None and qrs_max is not None:
        if qrs_min <= qrs_ms <= qrs_max:
            return (
                f"The QRS complex measured at {qrs_ms} ms falls within the "
                f"expected range ({qrs_min}\u2013{qrs_max} ms per the ontology)"
            )
        if qrs_ms < qrs_min:
            return (
                f"The QRS complex measured at {qrs_ms} ms is narrower than the "
                f"expected range ({qrs_min}\u2013{qrs_max} ms per the ontology)"
            )
        return (
            f"The QRS complex measured at {qrs_ms} ms exceeds the "
            f"expected range ({qrs_min}\u2013{qrs_max} ms per the ontology)"
        )
    return f"The QRS complex is measured at {qrs_ms} ms"


def _build_clinical_narrative(
    predicted_class: int, xai_features: dict, profile: dict
) -> str:
    """Generate a clinical narrative entirely from SPARQL-derived profile slots.

    Every clinical assertion (P-wave presence, QRS morphology, T-wave
    expected morphology) is sourced from the ontology ``profile`` dict
    produced by ``_sparql_beat_profile``.  Updating the OWL/Turtle data
    automatically updates the generated text — no Python code changes
    required.
    """
    focus = xai_features.get("focus_region", "indeterminate")
    qrs_ms = xai_features.get("qrs_duration_ms", 0.0)
    t_morph_measured = xai_features.get("t_wave_morphology", "indeterminate")

    # --- SPARQL-derived slots (auto-update when ontology changes) ---
    beat_label = profile.get("beat_label", _CLASS_TO_BEAT_NAME.get(predicted_class, "Unknown"))
    qrs_morph = profile.get("qrs_morph", "")
    p_presence = profile.get("p_presence", "")
    t_morph_expected = profile.get("t_morph_expected", "")

    # --- QRS assessment (already ontology-driven) ---
    qrs_text = _qrs_assessment(qrs_ms, profile.get("qrs_min"), profile.get("qrs_max"))

    # --- Compose narrative from slots ---
    parts: list[str] = []

    # 1) XAI attention region
    parts.append(
        f"The neural network focused its attention on the {focus} region."
    )

    # 2) QRS: duration check + ontology-defined morphology
    qrs_sentence = qrs_text
    if qrs_morph:
        qrs_sentence += f", with an expected morphology described as \"{qrs_morph}\" by the ontology"
    parts.append(f"{qrs_sentence}.")

    # 3) P-wave: presence description from ontology
    if p_presence:
        parts.append(
            f"The P-wave status for a {beat_label} is defined as \"{p_presence}\" "
            f"according to the ontological model."
        )

    # 4) T-wave: compare measured morphology against ontology expectation
    if t_morph_expected:
        parts.append(
            f"The measured T-wave morphology is {t_morph_measured}, "
            f"while the ontology expects \"{t_morph_expected}\" for this beat type."
        )
    else:
        parts.append(f"The measured T-wave morphology is {t_morph_measured}.")

    # 5) Conclusion
    parts.append(
        f"These neuro-symbolic findings support the classification as a {beat_label}."
    )

    return " ".join(parts)
