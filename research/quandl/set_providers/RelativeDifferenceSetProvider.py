import numpy as np


class RelativeDifferenceSetProvider:

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

    def examples_to_sets(self, examples, day_range):
        inputs = []
        outputs = []

        for exampleStartI in range(1, len(examples) - day_range - 1 + 1):
            input = []
            for exampleI in range(exampleStartI, exampleStartI + day_range):
                input.append(self._get_relative_difference(examples, exampleI))

            output = self._get_class_one_hot(
                self._get_relative_difference(examples, exampleStartI + day_range))

            inputs.append(input)
            outputs.append(output)

        inputs_np = np.array(inputs, dtype=np.float32)
        outputs_np = np.array(outputs, dtype=np.float32)

        return (inputs_np, outputs_np)

    def get_distribution(self, examples, day_range):
        (inputs, outputs) = self.examples_to_sets(examples, day_range)
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
                input.append([self._get_relative_difference(examples, exampleI)])

            output = self._get_class_one_hot(
                self._get_relative_difference(examples, exampleStartI + day_range))

            inputs.append(input)
            outputs.append(output)

        inputs_np = np.array(inputs, dtype=np.float32)
        outputs_np = np.array(outputs, dtype=np.float32)

        return (inputs_np, outputs_np)

    def examples_to_sets_raw(self, examples, day_range):
        inputs = []
        outputs = []

        for exampleStartI in range(1, len(examples) - day_range - 1 + 1):
            input = []
            for exampleI in range(exampleStartI, exampleStartI + day_range):
                input.append([self._get_relative_difference(examples, exampleI)])

            output = self._get_relative_difference(examples, exampleStartI + day_range)

            inputs.append(input)
            outputs.append(output)

        inputs_np = np.array(inputs, dtype=np.float32)
        outputs_np = np.array(outputs, dtype=np.float32)

        return (inputs_np, outputs_np)
