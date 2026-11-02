import hashlib
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from g3_dev_retention import score, token_f1


def test_reference_scoring_and_refusals():
    target = 'The red cat'
    manifest = {'requests': [{'id': 'x', 'split': 'dev'}]}
    selection = {'requests': [{'id': 'x', 'target_sha256': hashlib.sha256(target.encode()).hexdigest()}]}
    inputs = [{'id': 'x', 'target': target}]
    outcomes = [{'id': 'x', 'arm': 'full', 'output_text': 'red cat', 'complete': True},
                {'id': 'x', 'arm': 'reuse', 'output_text': 'red dog', 'complete': True}]
    assert token_f1('The red cat!', target) == 1
    result = score(manifest, selection, inputs, outcomes, ['x'])
    assert result['full_mean_f1'] == 1 and result['reuse_mean_f1'] == .5
    with pytest.raises(ValueError): score(manifest, selection, inputs, outcomes[:-1], ['x'])
    with pytest.raises(ValueError): score(manifest, selection, inputs, outcomes + outcomes[:1], ['x'])
    with pytest.raises(ValueError): score({'requests':[{'id':'x','split':'gate'}]}, selection, inputs, outcomes, ['x'])
    with pytest.raises(ValueError): score(manifest, selection, [{'id':'x','target':'changed'}], outcomes, ['x'])
    outcomes[1]['complete'] = False
    with pytest.raises(ValueError): score(manifest, selection, inputs, outcomes, ['x'])
