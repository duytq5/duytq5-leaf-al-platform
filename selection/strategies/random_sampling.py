from selection.base import LabeledStats, PoolData, Strategy, register


@register("random")
class RandomStrategy(Strategy):
    description = "Uniform random sample of the pool (baseline)."
    requires = frozenset()

    def select(self, pool: PoolData, labeled: LabeledStats, k: int) -> list[str]:
        k = min(k, len(pool))
        idx = self.rng.choice(len(pool), size=k, replace=False)
        return [pool.ids[i] for i in idx]
