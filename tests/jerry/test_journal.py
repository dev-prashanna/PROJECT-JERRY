import pytest
from jerry.journal import Journal


def test_admission_retries_and_restart_cannot_create_another_mission(tmp_path):
    store=Journal(tmp_path/'memory.db')
    payload=dict(session_id='session',epoch_id=0,map_revision=0,kind='explore')
    identity,status=store.admit('request',payload)
    assert status=='NEW'
    assert store.admit('request',payload)==(identity,'ADMITTED')
    with pytest.raises(ValueError):store.admit('request',{**payload,'kind':'navigate'})
    store.close()
    store=Journal(tmp_path/'memory.db')
    store.interrupt_previous()
    assert store.admit('request',payload)==(identity,'FAILED')
    store.close()


def test_terminal_states_and_explicit_facts_survive_reopening(tmp_path):
    path=tmp_path/'memory.db'
    store=Journal(path)
    identity,_=store.admit('task',dict(session_id='s',epoch_id=0,map_revision=0))
    store.transition(identity,'CANCELLED','operator stop')
    with pytest.raises(ValueError):store.transition(identity,'SUCCEEDED','late result')
    record=store.append('fact','explicit_fact','My chair is red')
    assert store.append('fact','explicit_fact','My chair is red')==record
    store.close()
    store=Journal(path)
    assert store.query('red')[0]['record_id']==record
    store.close()
