from aero_mesh.avl_reference import avl_geometry_text, run_avl_mesh_reference, run_avl_reference
from aero_mesh.geometry import generate_aero_mesh, make_wing_params_for_aspect_ratio


def test_avl_geometry_contains_symmetric_wing_sections():
    params = make_wing_params_for_aspect_ratio(8, sweep_deg=12, taper_ratio=0.6)
    text = avl_geometry_text(params)
    assert "SURFACE\nWing" in text
    assert "YDUPLICATE\n0.0" in text
    assert text.count("SECTION") == 2
    assert "AFILE\nsection-00.dat 0.0 1.0" in text
    assert "NACA" not in text


def test_official_avl_binary_when_available():
    executable = "scripts/run_avl_3_32.sh"
    params = make_wing_params_for_aspect_ratio(8, n_span=16, n_chord=4)
    result = run_avl_reference(params, 5.0, executable)
    assert 0.2 < result.cl < 0.8
    assert result.cdi > 0.0
    assert result.version == "3.32"


def test_structured_mesh_reference_matches_parameter_reference():
    executable = "scripts/run_avl_3_32.sh"
    params = make_wing_params_for_aspect_ratio(8, n_span=12, n_chord=4)
    parameter_result = run_avl_reference(params, 5.0, executable)
    mesh_result = run_avl_mesh_reference(generate_aero_mesh(params), 5.0, executable)
    assert abs(mesh_result.cl - parameter_result.cl) < 0.002
    assert abs(mesh_result.cdi - parameter_result.cdi) < 0.0002
    assert abs(mesh_result.cm - parameter_result.cm) < 0.0002
