from dataclasses import dataclass, field
from time import perf_counter
from contextlib import contextmanager
from collections import defaultdict

@dataclass
class RuntimeProfiler:
    stats: dict = field(default_factory=lambda: defaultdict(float))

    @contextmanager
    def measure(self, entity: str, op: str):
        start = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - start) * 1000.0
            self.stats[(entity, op)] += elapsed_ms

    def get_entity_total(self, entity: str) -> float:
        return sum(v for (e, _), v in self.stats.items() if e == entity)

    def to_dict(self):
        return dict(self.stats)

    def reset(self):
        self.stats.clear()
