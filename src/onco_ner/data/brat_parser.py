from pathlib import Path

from onco_ner.schemas import Entity, Span
from onco_ner.exceptions import InvalidAnnotationError
from onco_ner.utils.logging import get_logger

logger = get_logger(__name__)


def parse_ann_file(ann_path: Path) -> list[Entity]:
    """
    Parse un fichier .ann au format brat avec annotations ICD-O.

    Format attendu :
        T1\tmorphologie 2300 2311\tadénopathie
        T13\tmorphologie 3225 3234;3246 3258\tcarcinome microcytaire  (span discontinu)
        #1\tICD-O T1\t8000/1
    """
    raw_entities: dict[str, dict] = {}
    icdo_codes: dict[str, str] = {}

    with open(ann_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Première passe : entités (lignes T)
    for line_num, line in enumerate(lines, start=1):
        line = line.rstrip("\n")
        if not line:
            continue

        if line.startswith("T"):
            try:
                entity_id, span_info, text = line.split("\t", maxsplit=2)
                label, span_positions = span_info.split(" ", maxsplit=1)

                spans = []
                for sub_span in span_positions.split(";"):
                    start_str, end_str = sub_span.split(" ")
                    spans.append(Span(start=int(start_str), end=int(end_str)))

                if len(spans) > 1:
                    logger.info(
                        f"Span discontinu : {ann_path.name} ({entity_id}), "
                        f"{len(spans)} sous-spans"
                    )

                raw_entities[entity_id] = {
                    "id": entity_id,
                    "label": label,
                    "spans": tuple(spans),
                    "text": text,
                }
            except ValueError as e:
                raise InvalidAnnotationError(
                    f"Ligne T mal formée dans {ann_path.name} (ligne {line_num}): {line!r}"
                ) from e

    # Deuxième passe : normalisations ICD-O (lignes #)
    for line_num, line in enumerate(lines, start=1):
        line = line.rstrip("\n")
        if not line:
            continue

        if line.startswith("#"):
            try:
                _, ref_info, code = line.split("\t", maxsplit=2)
                ref_parts = ref_info.split(" ", maxsplit=1)
                if len(ref_parts) != 2:
                    raise InvalidAnnotationError(
                        f"Format de référence inattendu dans {ann_path.name} "
                        f"(ligne {line_num}): {ref_info!r}"
                    )
                _, entity_id = ref_parts
                if entity_id not in raw_entities:
                    raise InvalidAnnotationError(
                        f"Référence à une entité inconnue {entity_id} dans "
                        f"{ann_path.name} (ligne {line_num})"
                    )
                icdo_codes[entity_id] = code
            except ValueError as e:
                raise InvalidAnnotationError(
                    f"Ligne # mal formée dans {ann_path.name} (ligne {line_num}): {line!r}"
                ) from e

    # Construction finale : objets Entity immuables et complets dès la création
    entities = [
        Entity(**data, icdo_code=icdo_codes.get(entity_id))
        for entity_id, data in raw_entities.items()
    ]

    logger.info(f"{ann_path.name} : {len(entities)} entités parsées")
    return entities


def parse_document(txt_path: Path, ann_path: Path) -> dict:
    """Parse une paire .txt/.ann et retourne le texte + ses entités."""
    text = txt_path.read_text(encoding="utf-8")
    entities = parse_ann_file(ann_path)
    return {"doc_name": txt_path.stem, "text": text, "entities": entities}


def parse_corpus(data_dir: Path) -> list[dict]:
    """Parse l'ensemble du corpus FRACCO (.txt/.ann appariés)."""
    txt_files = sorted(data_dir.glob("*.txt"))
    documents = []

    for txt_path in txt_files:
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            logger.warning(f"Pas de fichier .ann pour {txt_path.name}, ignoré")
            continue
        documents.append(parse_document(txt_path, ann_path))

    logger.info(f"{len(documents)} documents parsés sur {len(txt_files)} fichiers .txt")
    return documents