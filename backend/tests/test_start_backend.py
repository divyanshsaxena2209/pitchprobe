from io import StringIO
from threading import Event

from start_backend import PORT_MAX, PORT_START, next_candidate_port, relay_child_output


def test_failed_candidate_always_advances_to_next_port():
    assert next_candidate_port(8000) == 8001
    assert next_candidate_port(8004) == 8005


def test_port_scan_wraps_only_after_maximum_port():
    assert next_candidate_port(PORT_MAX) == PORT_START


def test_child_listener_marker_must_match_attempted_port(capsys):
    wrong_port_event = Event()
    relay_child_output(
        StringIO(
            "INFO: Uvicorn running on http://127.0.0.1:8004 (Press CTRL+C to quit)\n"
        ),
        8005,
        wrong_port_event,
    )
    assert not wrong_port_event.is_set()

    own_port_event = Event()
    relay_child_output(
        StringIO(
            "INFO: Uvicorn running on http://127.0.0.1:8005 (Press CTRL+C to quit)\n"
        ),
        8005,
        own_port_event,
    )
    assert own_port_event.is_set()
