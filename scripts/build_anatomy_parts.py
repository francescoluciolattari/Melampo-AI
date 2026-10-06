"""Build data/linking/anatomy_parts.json: curated part-of knowledge for the anatomy linker.

Run from the repository root:  python scripts/build_anatomy_parts.py

Three kinds of knowledge, all written by hand from anatomy, none derived from the
test sets:

* ``direct``: a whole expression that names a part (or the outline, or a synonym) of a
  TotalSegmentator class: "sigma", "lingula", "profilo cardiaco". The side, when the
  class exists on both sides, is taken from the mention.
* ``parts``: a part word that combines with the name of an organ it belongs to:
  "testa" + "pancreas", "polo" + "rene" + side. Only the listed wholes are allowed:
  "testa del fegato" is not linked.
* relation: ``equal`` (synonym), ``part_of``, ``contour_of`` (the outline of the organ on
  a radiograph), ``approx`` (a usual radiological synonym that is not strictly identical).

Nothing here is a guess. A mention that is not in the table is not linked by it.
"""

import json
from pathlib import Path

direct = []


def add(whole, relation, *names):
    direct.append({"whole": whole, "relation": relation, "names": list(names)})


# Stomach, bowel
add(
    "stomach",
    "part_of",
    "fondo gastrico",
    "fundus gastrico",
    "gastric fundus",
    "fundus of the stomach",
    "antro gastrico",
    "antro pilorico",
    "gastric antrum",
    "pyloric antrum",
    "piloro",
    "pylorus",
    "canale pilorico",
    "cardias",
    "gastric cardia",
    "piccola curvatura gastrica",
    "grande curvatura gastrica",
)
add(
    "duodenum",
    "part_of",
    "bulbo duodenale",
    "duodenal bulb",
    "cornice duodenale",
    "ansa duodenale",
    "duodenal loop",
    "duodenal c-loop",
    "prima porzione duodenale",
    "seconda porzione duodenale",
    "terza porzione duodenale",
    "quarta porzione duodenale",
    "second part of the duodenum",
    "third part of the duodenum",
    "descending duodenum",
    "duodeno discendente",
    "duodeno orizzontale",
    "horizontal duodenum",
)
add(
    "small_bowel",
    "part_of",
    "ileum",
    "ileo terminale",
    "terminal ileum",
    "jejunum",
    "anse ileali",
    "anse digiunali",
    "anse del tenue",
    "ileal loops",
    "jejunal loops",
    "small bowel loops",
    "loops of small bowel",
    "anse del piccolo intestino",
)
add(
    "colon",
    "part_of",
    "ceco",
    "cecum",
    "caecum",
    "colon ascendente",
    "ascending colon",
    "colon trasverso",
    "transverse colon",
    "colon discendente",
    "descending colon",
    "sigma",
    "colon sigmoideo",
    "sigmoide",
    "sigmoid colon",
    "flessura epatica",
    "hepatic flexure",
    "flessura splenica",
    "splenic flexure",
    "parete colica",
    "colonic wall",
    "wall of the colon",
    "cornice colica",
)

# Liver and biliary tree, pancreas, spleen, kidney, bladder, prostate
add(
    "liver",
    "part_of",
    "cupola epatica",
    "hepatic dome",
    "dome of the liver",
    "lobo epatico destro",
    "lobo epatico sinistro",
    "right hepatic lobe",
    "left hepatic lobe",
    "right lobe of the liver",
    "left lobe of the liver",
)
add(
    "pancreas",
    "part_of",
    "processo uncinato",
    "uncinate process",
    "istmo pancreatico",
    "pancreatic neck",
    "collo del pancreas",
    "pancreatic head",
    "pancreatic body",
    "pancreatic tail",
    "testa del pancreas",
    "coda del pancreas",
    "corpo del pancreas",
)
add(
    "kidney",
    "part_of",
    "corticale renale",
    "renal cortex",
    "midollare renale",
    "renal medulla",
    "piramidi renali",
    "renal pyramids",
    "polo superiore del rene",
    "polo inferiore del rene",
)
add(
    "urinary_bladder",
    "part_of",
    "cupola vescicale",
    "bladder dome",
    "trigono vescicale",
    "bladder trigone",
    "collo vescicale",
    "bladder neck",
    "parete vescicale",
    "bladder wall",
)
add(
    "prostate",
    "part_of",
    "zona periferica prostatica",
    "peripheral zone of the prostate",
    "prostatic peripheral zone",
    "prostatic transition zone",
    "zona di transizione prostatica",
    "transition zone of the prostate",
    "zona centrale prostatica",
    "central zone of the prostate",
    "apice prostatico",
    "prostatic apex",
    "base prostatica",
    "prostatic base",
    "ghiandola prostatica",
)

# Thorax
add(
    "lung_upper_lobe_left",
    "part_of",
    "lingula",
    "lingula of the left lung",
    "lingula polmonare",
    "segmento linguale",
    "lingular segment",
)
add(
    "lung_middle_lobe_right",
    "part_of",
    "segmento laterale del lobo medio",
    "segmento mediale del lobo medio",
    "lateral segment of the middle lobe",
    "medial segment of the middle lobe",
)
add(
    "heart",
    "part_of",
    "ventricolo sinistro",
    "ventricolo destro",
    "left ventricle",
    "right ventricle",
    "atrio sinistro",
    "atrio destro",
    "left atrium",
    "right atrium",
    "miocardio",
    "myocardium",
    "setto interventricolare",
    "interventricular septum",
    "setto interatriale",
    "interatrial septum",
    "apice cardiaco",
    "cardiac apex",
    "valvola aortica",
    "aortic valve",
    "valvola mitrale",
    "mitral valve",
    "camere cardiache",
    "cardiac chambers",
)
add(
    "heart",
    "contour_of",
    "profilo cardiaco",
    "silhouette cardiaca",
    "cardiac silhouette",
    "cardiac contour",
    "ombra cardiaca",
    "cardiac shadow",
    "cardiac outline",
    "contorno cardiaco",
)
add(
    "aorta",
    "part_of",
    "aorta ascendente",
    "ascending aorta",
    "arco aortico",
    "aortic arch",
    "arco dell'aorta",
    "aorta discendente",
    "descending aorta",
    "aorta toracica",
    "thoracic aorta",
    "aorta addominale",
    "abdominal aorta",
    "radice aortica",
    "aortic root",
    "aorta toracica discendente",
    "descending thoracic aorta",
    "aorta sottorenale",
    "infrarenal aorta",
    "aorta sovrarenale",
    "aorta toraco-addominale",
)
add(
    "portal_vein_and_splenic_vein",
    "part_of",
    "vena porta principale",
    "main portal vein",
    "tronco portale",
    "ramo portale destro",
    "ramo portale sinistro",
    "right portal vein",
    "left portal vein",
    "right portal branch",
    "left portal branch",
)
add(
    "inferior_vena_cava",
    "part_of",
    "vena cava inferiore sottorenale",
    "infrarenal ivc",
    "infrarenal inferior vena cava",
    "vena cava inferiore sovraepatica",
    "suprahepatic ivc",
    "suprahepatic inferior vena cava",
    "vena cava inferiore intraepatica",
    "intrahepatic ivc",
)
add(
    "thyroid_gland",
    "part_of",
    "lobo tiroideo",
    "istmo tiroideo",
    "thyroid isthmus",
    "thyroid lobe",
)
add(
    "trachea",
    "part_of",
    "carena",
    "carina",
    "carena tracheale",
    "tracheal carina",
    "trachea cervicale",
    "trachea toracica",
    "cervical trachea",
    "thoracic trachea",
)
add(
    "esophagus",
    "part_of",
    "esofago cervicale",
    "esofago toracico",
    "esofago addominale",
    "esofago distale",
    "esofago prossimale",
    "esofago medio",
    "cervical esophagus",
    "thoracic esophagus",
    "abdominal esophagus",
    "distal esophagus",
    "proximal esophagus",
    "mid esophagus",
    "esofago toracico distale",
    "lower esophagus",
)
add(
    "sternum",
    "part_of",
    "manubrio",
    "manubrio sternale",
    "manubrium",
    "manubrium of the sternum",
    "processo xifoideo",
    "xifoide",
    "xiphoid process",
    "xiphoid",
    "corpo sternale",
    "sternal body",
)

# Spine, skull
add(
    "vertebrae_C2",
    "part_of",
    "processo odontoideo",
    "dente dell'epistrofeo",
    "odontoid process",
    "dens",
    "odontoid peg",
)
add(
    "sacrum",
    "part_of",
    "ala sacrale",
    "ala del sacro",
    "sacral ala",
    "ala of the sacrum",
    "promontorio sacrale",
    "sacral promontory",
)
add(
    "spinal_cord",
    "part_of",
    "midollo cervicale",
    "midollo toracico",
    "midollo lombare",
    "cono midollare",
    "conus medullaris",
    "cervical cord",
    "thoracic cord",
    "cervical spinal cord",
    "thoracic spinal cord",
    "lumbar spinal cord",
)
add("skull", "part_of", "teca cranica", "calvarium", "calvaria")
add(
    "skull",
    "part_of",
    "calotta cranica",
    "volta cranica",
    "cranial vault",
    "skull vault",
    "base cranica",
    "skull base",
    "base del cranio",
)
add(
    "brain",
    "part_of",
    "lobo frontale",
    "lobo parietale",
    "lobo temporale",
    "lobo occipitale",
    "frontal lobe",
    "parietal lobe",
    "temporal lobe",
    "occipital lobe",
    "cervelletto",
    "cerebellum",
    "tronco encefalico",
    "brainstem",
    "brain stem",
    "emisfero cerebrale",
    "cerebral hemisphere",
    "pons",
    "mesencefalo",
    "midbrain",
    "bulbo encefalico",
    "medulla oblongata",
)

# Sided families (the side comes from the mention)
add("hip", "approx", "emibacino", "emipelvi", "hemipelvis")
add("hip", "equal", "osso iliaco", "iliac bone")
add(
    "hip",
    "part_of",
    "ala iliaca",
    "iliac wing",
    "ala dell'ileo",
    "acetabolo",
    "acetabulum",
    "branca ischiopubica",
    "ischiopubic ramus",
    "branca ilio-pubica",
    "tuberosita ischiatica",
    "ischial tuberosity",
    "cresta iliaca",
    "iliac crest",
    "ischio",
    "ischium",
    "pube",
    "pubis",
)
add(
    "iliopsoas",
    "part_of",
    "psoas",
    "psoas maggiore",
    "muscolo psoas",
    "psoas muscle",
    "muscolo psoas maggiore",
    "psoas major muscle",
    "iliacus muscle",
    "iliac muscle",
    "muscolo iliaco",
    "muscolo ileopsoas",
    "psoas major",
    "muscolo iliaco",
    "iliacus",
)
add(
    "femur",
    "part_of",
    "testa femorale",
    "femoral head",
    "collo femorale",
    "femoral neck",
    "grande trocantere",
    "greater trochanter",
    "piccolo trocantere",
    "lesser trochanter",
    "diafisi femorale",
    "femoral shaft",
    "femoral diaphysis",
    "condili femorali",
    "femoral condyles",
    "femore prossimale",
    "proximal femur",
    "femore distale",
    "distal femur",
)
add(
    "humerus",
    "part_of",
    "testa omerale",
    "humeral head",
    "collo chirurgico dell'omero",
    "surgical neck of the humerus",
    "diafisi omerale",
    "humeral shaft",
    "humeral diaphysis",
    "omero prossimale",
    "proximal humerus",
    "omero distale",
    "distal humerus",
    "troclea omerale",
)
add(
    "scapula",
    "part_of",
    "glena",
    "glenoid",
    "cavita glenoidea",
    "acromion",
    "processo coracoideo",
    "coracoid process",
    "angolo inferiore della scapola",
    "inferior angle of the scapula",
    "spina della scapola",
    "scapular spine",
)
add(
    "clavicula",
    "part_of",
    "estremita acromiale",
    "acromial end of the clavicle",
    "estremita sternale",
    "sternal end of the clavicle",
    "diafisi clavicolare",
    "clavicular shaft",
)
add(
    "adrenal_gland",
    "part_of",
    "branca surrenalica",
    "adrenal limb",
    "corticale surrenalica",
    "adrenal cortex",
)

# "S3 epatico": the liver is named in the mention itself, so the level code is a Couinaud segment.
for n in range(1, 9):
    add(
        f"liver_segment_{n}",
        "equal",
        f"S{n} epatico",
        f"S{n} hepatic",
        f"hepatic S{n}",
        f"liver S{n}",
    )

# Part words that combine with the name of a whole. "wholes" are class ids or families (without _left/_right).
parts = [
    {
        "words": ["testa", "head"],
        "wholes": ["pancreas", "femur", "humerus"],
        "relation": "part_of",
    },
    {"words": ["coda", "tail"], "wholes": ["pancreas"], "relation": "part_of"},
    {"words": ["polo", "pole"], "wholes": ["kidney", "spleen"], "relation": "part_of"},
    {
        "words": ["lobo", "lobe"],
        "wholes": ["liver", "thyroid_gland"],
        "relation": "part_of",
    },
    {
        "words": ["collo", "neck"],
        "wholes": ["femur", "humerus", "gallbladder", "urinary_bladder"],
        "relation": "part_of",
    },
    {
        "words": ["fondo", "fundus"],
        "wholes": ["stomach", "gallbladder", "urinary_bladder"],
        "relation": "part_of",
    },
    {
        "words": ["diafisi", "shaft", "diaphysis"],
        "wholes": ["femur", "humerus", "clavicula"],
        "relation": "part_of",
    },
    {
        "words": ["istmo", "isthmus"],
        "wholes": ["thyroid_gland", "pancreas"],
        "relation": "part_of",
    },
    {
        "words": ["apice", "apex"],
        "wholes": ["heart", "prostate"],
        "relation": "part_of",
    },
    {
        "words": ["cupola", "dome"],
        "wholes": ["liver", "urinary_bladder"],
        "relation": "part_of",
    },
    {
        "words": ["distale", "distal"],
        "wholes": ["esophagus", "femur", "humerus", "clavicula"],
        "relation": "part_of",
    },
    {
        "words": ["prossimale", "proximal"],
        "wholes": ["esophagus", "femur", "humerus", "clavicula"],
        "relation": "part_of",
    },
    {
        "words": ["cervicale", "cervical"],
        "wholes": ["esophagus", "trachea"],
        "relation": "part_of",
    },
    {
        "words": ["toracico", "toracica", "thoracic"],
        "wholes": ["esophagus", "trachea", "aorta"],
        "relation": "part_of",
    },
    {
        "words": ["addominale", "abdominal"],
        "wholes": ["esophagus", "aorta"],
        "relation": "part_of",
    },
    {
        "words": ["ascendente", "ascending"],
        "wholes": ["aorta", "colon"],
        "relation": "part_of",
    },
    {
        "words": ["discendente", "descending"],
        "wholes": ["aorta", "colon"],
        "relation": "part_of",
    },
    {"words": ["trasverso", "transverse"], "wholes": ["colon"], "relation": "part_of"},
    {"words": ["emisfero", "hemisphere"], "wholes": ["brain"], "relation": "part_of"},
    {
        "words": ["ramo", "branch"],
        "wholes": ["portal_vein_and_splenic_vein"],
        "relation": "part_of",
    },
]

# Names that look like a class and are not. Linked by nothing: the linker abstains with the reason.
never = [
    {
        "reason": "hilum_is_not_the_organ_or_its_vessel",
        "names": [
            "porta hepatis",
            "hilum hepatis",
            "hepatic hilum",
            "ilo epatico",
            "ilo renale",
            "renal hilum",
            "ilo splenico",
            "splenic hilum",
            "ilo polmonare",
            "pulmonary hilum",
            "hilar region",
        ],
    },
    {
        "reason": "the_canal_holds_the_cord_it_is_not_the_cord",
        "names": [
            "canale midollare",
            "canale vertebrale",
            "canale rachideo",
            "spinal canal",
            "vertebral canal",
            "central canal",
            "canale spinale",
        ],
    },
]

never.append(
    {
        "reason": "ambiguous_word_in_italian_or_english",
        "names": [
            "ponte",
            "ponte osseo",
            "ileo",
            "dente",
            "midollo",
            "digiuno",
            "colica",
            "apice polmonare",
            "bulbo",
            "seno",
            "corpo",
            "testa",
            "base",
            "lobo",
            "polo",
        ],
    }
)

out = {
    "note": "Conoscenza parte->intero scritta a mano da anatomia, non ricavata dai set di test. "
    "relation: equal = sinonimo, part_of = parte dell'intero, contour_of = profilo radiografico dell'organo, "
    "approx = sinonimo radiologico d'uso non strettamente identico. Una menzione non presente non e collegata da questa tabella.",
    "direct": direct,
    "parts": parts,
    "never": never,
}
path = Path("data/linking/anatomy_parts.json")
path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("direct entries", sum(len(d["names"]) for d in direct), "part words", len(parts))
