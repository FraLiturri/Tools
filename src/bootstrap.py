import numpy as np


def mean(data: np.ndarray, function: callable = lambda x: x**2) -> float:
    return np.mean(function(data))


def checker(x: float, y: float) -> bool:
    return True if max(x, y) / min(x, y) >= 100 else False


class Bootstrap:
    def __init__(
        self,
        data: np.ndarray,
        *,
        blocks: int,
        boot_samples: int,
        therm: int = 0,
        function: callable = lambda x: x**2,
        primary_functions: list[callable] = [],
    ):
        data = data[therm:]  # discard initial therm steps
        self.data = np.asarray(data)
        self.k = blocks
        self.r = boot_samples
        self.n = len(self.data)
        self.function = function
        self.primary_functions = primary_functions

        try:
            assert checker(self.n, self.k)
        except AssertionError:
            print(
                f"\nWarning: Number of blocks k={self.k} is too large for data size n={self.n} "
                "(for a good bootstrap k << n is suggested)."
            )

        self.blocks = np.array_split(self.data, self.k)

    def execute(self) -> tuple[float, float]:

        if self.primary_functions != []:
            """Calculating mean:"""
            func_means = []
            for func in self.primary_functions:
                func_means.append(mean(self.data, function=func))
                
            estimated_mean = self.function(*func_means)

            """Calculating std:"""
            means = []
            F_val = []
            for _ in range(self.r):
                sampled_blocks = []
                mean_val = [] 
                for func in self.primary_functions:
                    sample = []
                    for _ in range(self.k):
                        random_index = np.random.randint(0, self.k)
                        sample.append(self.blocks[random_index])

                    sample = np.concatenate(sample)
                    sampled_blocks.append(sample)
                    block_mean = mean(sample, function=func)
                    mean_val.append(block_mean)
                means.append(mean_val)

            for i in range(len(means)): 
                mean_F = self.function(*means[i])
                F_val.append(mean_F)
            
            estimated_std = np.std(F_val, ddof=1)
            return estimated_mean, estimated_std

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
