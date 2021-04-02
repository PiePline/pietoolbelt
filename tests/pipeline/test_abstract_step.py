import unittest

__all__ = ['AbstractStepTest']

from typing import List

from pietoolbelt.pipeline.abstract_step import AbstractStepResult, AbstractStep


class AbstractStepTest(unittest.TestCase):
    class _StepMock(AbstractStepResult):
        def get_output_paths(self) -> List[str]:
            return ['test1', 'test2']

    def test_result(self):
        self.assertEqual(AbstractStepTest._StepMock().get_output_paths(), ['test1', 'test2'])

    def test_step(self):
        class StepMock(AbstractStep):
            def __init__(self, output_res: AbstractStepResult, input_results: List[AbstractStepResult]):
                super().__init__(output_res, input_results=input_results)
                self.result = None

            def run(self):
                self.result = 'result'

        output_res = AbstractStepTest._StepMock()
        input_results = [AbstractStepTest._StepMock()]
        step = StepMock(output_res=output_res, input_results=input_results)
        step.run()

        self.assertEqual(step.result, 'result')
        self.assertIs(step.get_output_res(), output_res)

        for i, inp_res in enumerate(step.get_input_results()):
            self.assertIs(inp_res, input_results[i])
