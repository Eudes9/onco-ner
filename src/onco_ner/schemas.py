from pydantic import BaseModel, Field


class Span(BaseModel):
    model_config = {"frozen": True}

    start: int = Field(ge=0)
    end: int = Field(ge=0)


class Entity(BaseModel):
    model_config = {"frozen": True}

    id: str
    label: str
    spans: tuple[Span, ...]   # tuple (pas list) pour rester hachable
    text: str
    icdo_code: str | None = None

    @property
    def is_discontinuous(self) -> bool:
        return len(self.spans) > 1

    @property
    def start(self) -> int:
        """Position de début du premier sous-span."""
        return self.spans[0].start

    @property
    def end(self) -> int:
        """Position de fin du dernier sous-span."""
        return self.spans[-1].end