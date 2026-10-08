"""History persistence flag must preserve existing files and result logging."""
from types import SimpleNamespace as NS

import pytest

from icir_cleanroom.gas_mapping.config import ControllerConfig
from icir_cleanroom.gas_mapping.history.repository import GasHistoryStore
from icir_cleanroom.gas_mapping.ros.controller_node import GasMappingControllerNode
from icir_cleanroom.gas_mapping.application.hrs_run_log import HrsRunLog


def controller(store, enabled):
    return NS(history=store, save_history=enabled,
              get_logger=lambda: NS(info=lambda msg: None, debug=lambda msg: None),
              publish_history=lambda: None)


def test_disabled_saves_preserve_file_and_csv_still_works(tmp_path, map_message_factory):
    path = tmp_path/'history.json'
    store = GasHistoryStore(path, map_message_factory(), .2)
    store.save()
    original = path.read_bytes()
    assert store.load()  # Loading remains available, independent of save flag.
    store.history_count = 7
    c = controller(store, False)
    assert not GasMappingControllerNode.persist_history(c, 'test')
    assert path.read_bytes() == original
    response = GasMappingControllerNode.clear_history(c, None, NS())
    assert response.success and store.history_count == 0
    assert path.read_bytes() == original
    log = HrsRunLog(tmp_path/'results', 'test_clock')
    log.start(0., 0., event_id='test', lrs_lap=1, source=None,
              parameters={'save_history': False})
    log.finish(1., 1., reason='test', robot_xy=None, source_end=None)
    assert (log.directory/'runs.csv').is_file()
    assert (log.directory/'events.csv').is_file()


def test_enabled_saves_write_history(tmp_path, map_message_factory):
    path = tmp_path/'history.json'
    store = GasHistoryStore(path, map_message_factory(), .2)
    c = controller(store, True)
    assert GasMappingControllerNode.persist_history(c, 'test')
    assert path.is_file()
    assert GasMappingControllerNode.clear_history(c, None, NS()).success


def test_save_history_boolean_config():
    values = ControllerConfig.defaults().flat_values()
    assert values['save_history'] is True
    values['save_history'] = False
    assert ControllerConfig.from_mapping(values).history.save_history is False
    values['save_history'] = 'false'
    with pytest.raises(ValueError, match='save_history'):
        ControllerConfig.from_mapping(values)


def test_disabled_history_loading_does_not_read_existing_file(tmp_path, map_message_factory, monkeypatch):
    path = tmp_path / 'history.json'
    path.write_text('This must never be read or parsed.')
    store = GasHistoryStore(path, map_message_factory(), .2)
    store.history_count = 7
    calls = []
    monkeypatch.setattr(store, 'load', lambda: calls.append('load'))
    c = NS(history=store, load_history=False,
           get_logger=lambda: NS(info=lambda message: None))
    GasMappingControllerNode.restore_history(c)
    assert calls == []
    assert store.history_count == 0 and store.records == {}
    assert path.read_text() == 'This must never be read or parsed.'


def test_load_history_boolean_config():
    values = ControllerConfig.defaults().flat_values()
    assert values['load_history'] is True
    values['load_history'] = False
    assert ControllerConfig.from_mapping(values).history.load_history is False
    values['load_history'] = 'false'
    with pytest.raises(ValueError, match='load_history'):
        ControllerConfig.from_mapping(values)
