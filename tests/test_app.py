from streamlit.testing.v1 import AppTest


def test_streamlit_app_runs_without_exception():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert any("Coût annuel estimé" in m.label for m in at.metric)


def test_prediction_reacts_to_smoking():
    at = AppTest.from_file("app/streamlit_app.py", default_timeout=120).run()
    base = at.metric[0].value
    smoker_radio = next(r for r in at.radio if r.label == "Fumeur")
    smoker_radio.set_value("yes")
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value != base
