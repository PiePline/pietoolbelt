import os
import shutil
import unittest

__all__ = ['AbstractStepTest']

from typing import List

from pietoolbelt.pipeline.abstract_step import AbstractStepResult, AbstractStep, AbstractStepDirResult


class AbstractStepTest(unittest.TestCase):
    class _StepResultMock(AbstractStepResult):
        def get_output_paths(self) -> List[str]:
            return ['test1', 'test2']

    class _StepDirResultMock(AbstractStepDirResult):
        def __init__(self, path: str):
            super().__init__(path)

    RESULT_DIR = 'folded_train_result'

    def tearDown(self) -> None:
        if os.path.exists(AbstractStepTest.RESULT_DIR):
            shutil.rmtree(AbstractStepTest.RESULT_DIR, ignore_errors=True)

    def test_result(self):
        self.assertEqual(AbstractStepTest._StepResultMock().get_output_paths(), ['test1', 'test2'])

    def test_step_dirt_result(self):
        try:
            AbstractStepTest._StepDirResultMock(AbstractStepTest.RESULT_DIR)
            AbstractStepTest._StepDirResultMock(AbstractStepTest.RESULT_DIR)
        except Exception as err:
            self.fail("Error while instantiate AbstractDirResult. Error: [{}]".format(err))

    def test_step(self):
        class StepMock(AbstractStep):
            def __init__(self, output_res: AbstractStepResult, input_results: List[AbstractStepResult]):
                super().__init__(output_res, input_results=input_results)
                self.result = None

            def run(self):
                self.result = 'result'

        output_res = AbstractStepTest._StepResultMock()
        input_results = [AbstractStepTest._StepResultMock()]
        step = StepMock(output_res=output_res, input_results=input_results)
        step.run()

        self.assertEqual(step.result, 'result')
        self.assertIs(step.get_output_res(), output_res)

        for i, inp_res in enumerate(step.get_input_results()):
            self.assertIs(inp_res, input_results[i])
