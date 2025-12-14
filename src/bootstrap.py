import numpy as np


def mean(data: np.ndarray, function: callable = lambda x: x**2) -> float:
    return np.mean(function(data))


def checker(x: float, y: float) -> bool:
    # returns True if x is significantly greater than y (or viceversa)
    if x >= y:
        return x / 100 >= y
    else:
        return y / 100 >= x


class Bootstrap:
    def __init__(
        self,
        data: np.ndarray,
        *,
        blocks: int,
        boot_samples: int,
        function: callable = lambda x: x**2,
    ):
        self.data = np.asarray(data)
        self.k = blocks
        self.r = boot_samples
        self.n = len(self.data)
        self.function = function

        try:
            assert checker(self.n, self.k)
        except AssertionError:
            print(
                f"\nWarning: Number of blocks k={self.k} is too large for data size n={self.n} "
                "(for a good bootstrap k << n is suggested)."
            )

        self.blocks = np.array_split(self.data, self.k)

    def execute(self) -> tuple[float, float]:
        means = []
        for _ in range(self.r):
            sampled_blocks = []
            for _ in range(self.k):
                random_index = np.random.randint(0, self.k)
                sampled_blocks.append(self.blocks[random_index])

            sampled_blocks = np.concatenate(sampled_blocks)
            block_mean = mean(sampled_blocks, function=self.function)
            means.append(block_mean)

        means = np.array(means)
        estimated_mean = mean(self.data, function=self.function)
        estimated_std = np.std(means, ddof=1)

        return estimated_mean, estimated_std 

    def __call__(self):
        return self.execute()
