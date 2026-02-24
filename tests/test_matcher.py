import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from src.template_matcher import TemplateMatcher, TemplateMatch
from src.algorithm_classifier import AlgorithmInfo


@pytest.fixture
def matcher():
    m = TemplateMatcher()
    m.load()
    return m


def test_matcher_loads(matcher):
    assert matcher._loaded


def test_search_returns_list(matcher):
    info = AlgorithmInfo(
        algo_type="gemm",
        confidence=0.8,
        parameters={"M": 4096, "N": 4096, "K": 4096},
        features={
            "mfma_count": 48,
            "mfma_types": {"v_mfma_f32_16x16x16_bf16": 48},
            "total_instructions": 500,
            "max_vgpr": 100,
            "dpp_count": 0,
            "vectorization_ratio": 0.5,
        },
        sub_type="gemm_bf16",
    )
    matches = matcher.search(info, arch="gfx942", top_k=3)
    assert isinstance(matches, list)
    for m in matches:
        assert isinstance(m, TemplateMatch)
        assert 0 <= m.similarity_score <= 1.0


def test_get_corpus_categories(matcher):
    cats = matcher.get_corpus_categories()
    assert isinstance(cats, dict)


def test_empty_search():
    m = TemplateMatcher()
    m.load()
    info = AlgorithmInfo(
        algo_type="nonexistent_type",
        confidence=0.1,
        parameters={},
        features={},
        sub_type="nonexistent",
    )
    matches = m.search(info, arch="gfx999")
    assert isinstance(matches, list)
