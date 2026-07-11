from __future__ import annotations

import json

import pytest

from multi_turboquant.integration import (
    LMCacheIntegrationConfig,
    build_lmcache_launch_plan,
)


def transfer_payload(plan) -> dict:
    assert plan.vllm_args[0] == "--kv-transfer-config"
    return json.loads(plan.vllm_args[1])


def test_multiprocess_plan_matches_documented_connector_contract():
    plan = build_lmcache_launch_plan(LMCacheIntegrationConfig())
    payload = transfer_payload(plan)
    assert payload == {
        "kv_connector": "LMCacheMPConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "lmcache.mp.host": "tcp://localhost",
            "lmcache.mp.port": 5555,
        },
    }
    assert plan.server_command == (
        "lmcache", "server", "--host", "localhost", "--port", "5555",
        "--l1-size-gb", "5", "--eviction-policy", "LRU", "--chunk-size", "256",
    )
    assert plan.environment == {}


def test_in_process_plan_uses_config_file_without_server_command():
    plan = build_lmcache_launch_plan(LMCacheIntegrationConfig(
        mode="in_process",
        config_file=" /opt/lmcache.yaml ",
    ))
    assert transfer_payload(plan) == {
        "kv_connector": "LMCacheConnectorV1",
        "kv_role": "kv_both",
    }
    assert plan.server_command is None
    assert plan.environment == {"LMCACHE_CONFIG_FILE": "/opt/lmcache.yaml"}


def test_lmcache_shipped_connector_is_version_gated():
    for version in ("0.19.2", "0.20.0rc1"):
        with pytest.raises(ValueError, match="0.20.0 or newer"):
            build_lmcache_launch_plan(LMCacheIntegrationConfig(
                use_lmcache_shipped_connector=True,
                vllm_version=version,
            ))

    plan = build_lmcache_launch_plan(LMCacheIntegrationConfig(
        use_lmcache_shipped_connector=True,
        vllm_version="v0.20.0",
    ))
    assert transfer_payload(plan)["kv_connector_module_path"] == (
        "lmcache.integration.vllm.lmcache_mp_connector"
    )


@pytest.mark.parametrize("kwargs, message", [
    ({"mode": "invalid"}, "Unknown LMCache mode"),
    ({"mp_port": 0}, "port must be between"),
    ({"mp_host": "udp://host"}, "tcp://"),
    ({"mp_host": "tcp://"}, "include a hostname"),
    ({"mp_host": "tcp://host:5555"}, "must not include a port"),
    ({"mp_host": "host/path"}, "without whitespace or a path"),
    ({"server_l1_size_gb": 0}, "L1 size must be positive"),
    ({"server_l1_size_gb": float("nan")}, "L1 size must be positive"),
    ({"server_chunk_size": 0}, "chunk size must be positive"),
    ({"server_eviction_policy": "random"}, "documented LRU"),
    ({"config_file": "   "}, "config file cannot be blank"),
])
def test_invalid_lmcache_config_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        build_lmcache_launch_plan(LMCacheIntegrationConfig(**kwargs))


def test_existing_kv_transfer_config_is_not_overwritten():
    plan = build_lmcache_launch_plan(LMCacheIntegrationConfig())
    command = plan.extend_vllm_command(["vllm", "serve", "model"])
    assert command[-2:] == list(plan.vllm_args)
    with pytest.raises(ValueError, match="already contains"):
        plan.extend_vllm_command([
            "vllm", "serve", "model", "--kv-transfer-config", "{}",
        ])
