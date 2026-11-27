import ast
import json
import re

import numpy as np
import pandas as pd
import pytest
from synthetic_data import NOTEBOOK_PATH


@pytest.fixture(scope="session")
def notebook_functions():
    """노트북 코드 셀의 최상위 함수 정의만 실행한 네임스페이스. 셀의 나머지 코드는 실행하지 않는다."""
    cells = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))["cells"]
    definitions = []
    for cell in cells:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            definitions += [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"np": np, "pd": pd, "re": re}
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(module, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace
