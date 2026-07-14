from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RCA_", extra="ignore")

    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "rca"
    pg_db: str = "rca"
    pg_password: str = ""

    topology_backend: str = "networkx"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    llm_enabled: bool = True
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "qwen2.5:3b"
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 1
    llm_temperature: float = 0.0

    data_root: Path = Path("./data")
    artifacts_root: Path = Path("./artifacts")

    use_autoencoder: bool = True
    use_deeplog: bool = True
    use_gnn: bool = True

    drain_sim_th: float = 0.4
    drain_depth: int = 4
    drain_max_children: int = 100
    iforest_contamination: float = 0.02
    iforest_n_estimators: int = 200
    eps_diag_alpha: float = 0.01
    pc_alpha: float = 0.05
    ppr_damping: float = 0.85
    ppr_max_iter: int = 100
    gnn_bias_weight: float = 1.0
    anomaly_zscore: float = 3.0
    dtw_radius: int = 10
    normal_window_frac: float = 0.5

    @computed_field
    @property
    def pg_dsn(self) -> str:
        return f"postgresql://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_db}"

    @computed_field
    @property
    def rcaeval_root(self) -> Path:
        return self.data_root / "rcaeval"

    @computed_field
    @property
    def unsw_root(self) -> Path:
        return self.data_root / "unsw_nb15"

    @computed_field
    @property
    def loghub_root(self) -> Path:
        return self.data_root / "loghub"


@lru_cache
def get_settings() -> Settings:
    return Settings()
