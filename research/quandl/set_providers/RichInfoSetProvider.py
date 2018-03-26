import numpy as np


class RichInfoSetProvider:

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

    def _get_relative_difference(self, examples, i, getter):
        return getter(examples[i]) / getter(examples[i - 1]) - 1

    def get_distribution(self, examples, day_range):
        (inputs, outputs) = self.examples_to_sets_arrays(examples, day_range)
        counts = [0, 0, 0]
        for i in range(0, len(outputs)):
            for i2 in range(0, len(counts)):
                counts[i2] += outputs[i][i2]

        for i2 in range(0, len(counts)):
            counts[i2] /= len(outputs)

        return counts

    def examples_to_sets_arrays(self, examples, day_range):
        inputs = []
        outputs = []

        for exampleStartI in range(1, len(examples) - day_range - 1 + 1):
            input = []
            for exampleI in range(exampleStartI, exampleStartI + day_range):
                input.append([
                    self._get_relative_difference(examples, exampleI, lambda x: x.adj_open),
                    self._get_relative_difference(examples, exampleI, lambda x: x.adj_high),
                    self._get_relative_difference(examples, exampleI, lambda x: x.adj_low),
                    self._get_relative_difference(examples, exampleI, lambda x: x.adj_close),
                    self._get_relative_difference(examples, exampleI, lambda x: x.adj_volume),
                    examples[exampleI].ex_dividend
                ])

            output = self._get_class_one_hot(
                self._get_relative_difference(examples, exampleStartI + day_range, lambda x: x.adj_close))

            inputs.append(input)
            outputs.append(output)

        inputs_np = np.array(inputs, dtype=np.float32)
        outputs_np = np.array(outputs, dtype=np.float32)

        return (inputs_np, outputs_np)