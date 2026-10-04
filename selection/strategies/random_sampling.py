from selection.base import PoolData, Strategy, register


@register("random")
class RandomStrategy(Strategy):
    requires = frozenset()

    def select(self, pool: PoolData, labeled: PoolData, k: int, **cfg) -> list[str]:
        k = min(k, len(pool))
        idx = self.rng.choice(len(pool), size=k, replace=False)
        return [pool.ids[i] for i in idx]
