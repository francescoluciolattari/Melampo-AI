"""Tests for the configurable FalkorDB connection: mode switching by editing
a config file only, verified against a real embedded instance -- never a mock.
"""

import tempfile
from pathlib import Path

import pytest

from melampo.memory.falkordb_connection import (
    MODE_LITE,
    MODE_SERVICE,
    FalkorDBConfig,
    connect,
)

# --------------------------------------------------------------------------
# Config file: created with a sensible default, read back correctly
# --------------------------------------------------------------------------


def test_a_missing_config_file_is_created_with_the_lite_default():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "falkordb_config.toml"
        assert not path.exists()

        config = FalkorDBConfig.from_file(path)

        assert path.exists()
        assert config.mode == MODE_LITE


def test_an_existing_lite_config_is_read_correctly():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "falkordb_config.toml"
        path.write_text('mode = "lite"\n[lite]\ndb_path = "somewhere/custom.db"\n')

        config = FalkorDBConfig.from_file(path)

        assert config.mode == MODE_LITE
        assert config.lite_db_path == "somewhere/custom.db"


def test_an_existing_service_config_is_read_correctly():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "falkordb_config.toml"
        path.write_text(
            'mode = "service"\n[service]\nhost = "graph.example.com"\nport = 12000\n'
            'password = "secret"\nssl = true\n'
        )

        config = FalkorDBConfig.from_file(path)

        assert config.mode == MODE_SERVICE
        assert config.service_host == "graph.example.com"
        assert config.service_port == 12000
        assert config.service_password == "secret"
        assert config.service_ssl is True


def test_an_unrecognised_mode_raises_clearly():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "falkordb_config.toml"
        path.write_text('mode = "something_else"\n')

        with pytest.raises(ValueError, match="unrecognised mode"):
            FalkorDBConfig.from_file(path)


def test_an_empty_password_in_the_file_becomes_none_not_an_empty_string():
    """A falkordb-py connection with password="" behaves differently from
    password=None in some Redis configurations -- keep the distinction."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "falkordb_config.toml"
        path.write_text('mode = "service"\n[service]\npassword = ""\n')

        config = FalkorDBConfig.from_file(path)

        assert config.service_password is None


# --------------------------------------------------------------------------
# connect(): a real embedded connection, real Cypher, real round trip --
# never mocked, since the whole point is that the interface genuinely works
# --------------------------------------------------------------------------


def test_connect_in_lite_mode_returns_a_working_connection():
    with tempfile.TemporaryDirectory() as directory:
        config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(Path(directory) / "test.db"))

        db = connect(config)
        graph = db.select_graph("melampo_test")
        graph.query('CREATE (m:Malattia {nome: "Marfan"})')
        result = graph.query("MATCH (m:Malattia) RETURN m.nome")

        assert result.result_set == [["Marfan"]]


def test_connect_in_lite_mode_creates_the_db_file_directory():
    with tempfile.TemporaryDirectory() as directory:
        nested_path = Path(directory) / "nested" / "subdir" / "test.db"
        config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(nested_path))

        connect(config)

        assert nested_path.parent.exists()


def test_a_cypher_relationship_round_trips_through_a_real_lite_connection():
    """Matches the exact scenario this module exists for: a disease-finding
    edge, queried back by traversal, not just a single node."""
    with tempfile.TemporaryDirectory() as directory:
        config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(Path(directory) / "test.db"))
        db = connect(config)
        graph = db.select_graph("melampo_test")

        graph.query(
            'CREATE (m:Malattia {nome: "Sindrome di Marfan"})'
            '-[:HA_FENOTIPO]->(f:Reperto {nome: "Aneurisma della radice aortica"})'
        )
        result = graph.query("MATCH (m:Malattia)-[:HA_FENOTIPO]->(f:Reperto) RETURN m.nome, f.nome")

        assert result.result_set == [["Sindrome di Marfan", "Aneurisma della radice aortica"]]
