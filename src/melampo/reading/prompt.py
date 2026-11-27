"""The one directive of the interpreters, for every text (25 November 2026).

The description of a finding is asked in the same words for a report and for a publication; only the part that
says what the text states about it changes (``card.STATEMENT``). The interpreters (Nemotron 3 Ultra and Muse Spark
Open, with reasoning) interpret; they give no code: codes come afterwards, to what both agree on.
"""

from __future__ import annotations

from . import card as c

DESCRIBE = (
    "- words: the words of the text;\n"
    "- name: its general name in English, the thing with its interpretation, without the values of one patient "
    "(\"elevated C-reactive protein\", \"right lower quadrant tenderness\", \"appendix wider than 6 mm\");\n"
    "- category: manifestation (what the patient has and an exam shows: a symptom, a sign, a laboratory value beyond "
    "a cut-off, an imaging, endoscopic, intraoperative or histological finding, a complication of the disease itself), "
    "test (an examination as a whole, CT, ultrasound, MRI, endoscopy), score (a clinical score or rule), outcome "
    "(death, a complication of treatment, recurrence), risk_factor (what makes a disease more likely, not what it "
    "shows), form (a subtype, stage or grade of a disease, such as gangrenous appendicitis);\n"
    "- specific: true when it is ONE thing; false when it is a list or a general phrase (\"typical symptoms\");\n"
    "- what: the thing observed, in English, without its interpretation (\"C-reactive protein\", \"abdominal "
    "tenderness\", \"appendix diameter\", \"fever\");\n"
    f"- interpretation: what the observation says of it, one of {', '.join(c.INTERPRETATIONS)} (present when the "
    "thing itself is the finding, such as fever or a mass), or \"\";\n"
    "- value: the value the text gives for it, {\"number\": n, \"unit\": \"...\"}, or null;\n"
    "- threshold: the cut-off the text gives, {\"comparator\": \">\", \">=\", \"<\", \"<=\" or \"=\", \"number\": n, "
    "\"unit\": \"...\"}, or null;\n"
    "- site: the body site as written, or \"\"; side: left, right, both, or \"\";\n"
    f"- observed_by: one of {', '.join(c.OBSERVED_BY)};\n"
    "- time: when it is observed or its order, as written (\"before the pain\", \"on the second day\"), or \"\";\n"
    "- nature: pathological, or physiological (a normal response, such as a raised heart rate with fever);\n"
    "- part: the part of the document its sentence is in, read on the whole document as its author wrote it, with "
    "or without a heading: in a report or a letter indication (the question asked, the reason for the exam, the "
    "referral, the clinical information), history, comparison (an earlier exam), technique (how the exam was done, "
    "the material or specimen examined), findings (what this exam or visit observes), impression (the conclusion, "
    "the diagnosis); in a publication background, methods, results, discussion; other for anything else;\n")

STATE = {
    "case": (
        "- state: present, absent (the text denies it: \"no fever\", \"not tender\"), or uncertain (\"possible\", "
        "\"cannot be excluded\", \"to be ruled out\");\n"
        "- temporality: current, past (history, a previous episode), or hypothetical (a plan, a risk);\n"
        "- subject: patient, or family_member (\"mother with diabetes\");\n"),
    "publication": (
        "- direction: when the text says, without a number, that having it makes the disease more likely or less "
        "likely: {\"when\": \"present\" or \"absent\", \"effect\": \"more_likely\" or \"less_likely\", \"words\": the "
        "words of strength as written, or \"\"}; null otherwise;\n"
        "- compares: what its measures compare: disease_vs_without, forms_of_the_disease (one form against "
        "another), other;\n"
        "- measures: null when the text gives no measure of it; otherwise ONE object holding every measure, as "
        "written, each field null when absent: with_disease {count, total, percent}, sensitivity, specificity, "
        "lr_positive, lr_negative, comparator (the patients without the disease the specificity was measured in), "
        "setting (emergency department, primary care, hospital, surgical, population, other), patients, design "
        "(meta_analysis, systematic_review, prospective_cohort, retrospective_cohort, cross_sectional, case_control, "
        "case_series, case_report, review, guideline, other), pooled (true for a meta-analysis's pooled value), verbal "
        "(a word of frequency such as \"often\" when there is no number); never a measure outside this object;\n"),
}

TASK = {
    "case": ("You read a document of one patient (a report, a note, a letter, in any language). List every finding "
             "it states about the patient or the family (symptoms, signs, laboratory values, imaging, endoscopic, "
             "intraoperative or histological findings), present, absent or uncertain, and also the exams, scores, "
             "risk factors and diseases it names, each as one object:\n"),
    "publication": ("You read a medical publication about {subject}. List everything it states that a patient with "
                    "{subject} shows or may show (symptoms, signs, laboratory values, imaging, endoscopic, "
                    "intraoperative or histological findings, complications of the disease itself), and also the "
                    "exams, scores, outcomes, risk factors and forms of the disease it measures or names, each as "
                    "one object:\n"),
}


def reading_prompt(context: str, sections: list[tuple[str, str]], subject: str = "") -> str:
    """The directive for one text: the same description for every context, the context's statement, the one form."""
    if context not in c.CONTEXTS:
        raise ValueError(f"context {context!r}: one of {c.CONTEXTS}")
    text = "\n\n".join(f"## {t}\n{x}" if t else x for t, x in sections)
    return (TASK[context].format(subject=subject or "the disease") + DESCRIBE + STATE[context]
            + "- quote: ONE sentence or ONE table row copied word for word from the text, that states it.\n"
            "Every field takes ONE value from its list (never two joined, never a value of your own). Each object "
            "has exactly these fields, in this form:\n" + c.example(context) + "\n"
            "Do not compute anything the text does not state, and give no code. Answer only a JSON list of such "
            "objects, [] when there is none.\n\n" + text)


__all__ = ["reading_prompt"]
