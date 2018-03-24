import numpy as np


class RawSetProvider:

    def _get_class_one_hot(self, difference):

        if difference < -0.0041:
            difference_class = 0
        elif difference > 0.0087:
            difference_class = 2
        else:
            difference_class = 1

        one_hot = [0, 0, 0]
        one_hot[difference_class] = 1
        return one_hot

    def _get_relative_difference(self, examples, i):
        return examples[i].adj_close / examples[i - 1].adj_close - 1

    def _normalize_sequence(self, sequence):
        min_value = min(sequence)
        max_value = max(sequence)

        norm_sequence = list(map(lambda v: (v - min_value) / (max_value - min_value), sequence))

        return norm_sequence

    def examples_to_sets(self, examples, day_range):
        inputs = []
        outputs = []

        for exampleStartI in range(1, len(examples) - day_range - 1):
            input = []
            for exampleI in range(exampleStartI, exampleStartI + day_range):
                input.append(examples[exampleI].adj_close)

            input = self._normalize_sequence(input)
            output = self._get_class_one_hot(
                self._get_relative_difference(examples, exampleStartI + day_range + 1))

            inputs.append(input)
            outputs.append(output)

        inputs_np = np.array(inputs, dtype=np.float32)
        outputs_np = np.array(outputs, dtype=np.float32)

        return (inputs_np, outputs_np)