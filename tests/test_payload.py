from src.workflows.generate import GenerateParams


def test_default_cfg_scale():
	p = GenerateParams(mode="t2i", prompt="test", model="flash")
	assert p.cfg_scale == 1.0

	p2 = GenerateParams(mode="t2i", prompt="test", model="turbo")
	assert p2.cfg_scale == 1.0

	p3 = GenerateParams(mode="t2i", prompt="test", model="large")
	assert p3.cfg_scale == 4.0

	p4 = GenerateParams(mode="t2i", prompt="test", model="medium")
	assert p4.cfg_scale == 4.0

	p5 = GenerateParams(mode="t2i", prompt="test", model="flash", cfg_scale=2.5)
	assert p5.cfg_scale == 2.5
