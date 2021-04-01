import json
import os
from typing import List


class StepMeta:
    def __init__(self, path: str):
        self._path = path
        self._steps = dict()

        if os.path.exists(path):
            with open(self._path, 'r') as meta_file:
                self._steps = json.load(meta_file)

    def update_step(self, path: str, status: str) -> 'StepMeta':
        if path in self._steps:
            self._steps[path]['status'] = status
        else:
            self._steps[path] = {'status': status}

        with open(self._path, 'w') as meta_file:
            json.dump(self._steps, meta_file, indent=4)

        return self

    def get_steps(self, status: str = 'any') -> List[str]:
        if status == 'any':
            return list(self._steps.keys())

        return [k for k, v in self._steps.items() if v['status'] == status]
