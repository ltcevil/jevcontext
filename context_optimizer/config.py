"""Isolated Configuration Manager for Context Optimizer.

Ensures the script uses its own isolated environment variables and settings,
without reading from or polluting global ambient environment variables.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


DEFAULT_CONFIG_DIR = Path.home() / ".config" / "context-optimizer"
DEFAULT_ENV_FILE = DEFAULT_CONFIG_DIR / "env"
LOCAL_ENV_FILE = Path(".env.context_optimizer")
OPTIMIZER_ENV_FILE = Path(".optimizer.env")


def resolve_adc_project() -> Optional[str]:
    """Resolve default GCP project ID from Google Cloud Application Default Credentials."""
    adc_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    candidate_paths: List[Path] = []
    if adc_env:
        candidate_paths.append(Path(os.path.expandvars(os.path.expanduser(adc_env))))
    candidate_paths.append(Path.home() / ".config" / "gcloud" / "application_default_credentials.json")

    for path in candidate_paths:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("quota_project_id"):
                    return str(data["quota_project_id"])
                if data.get("project_id"):
                    return str(data["project_id"])
            except Exception:
                pass

    try:
        import google.auth
        creds, proj = google.auth.default()
        if hasattr(creds, "quota_project_id") and creds.quota_project_id:
            return str(creds.quota_project_id)
        if hasattr(creds, "project_id") and creds.project_id:
            return str(creds.project_id)
        if proj:
            return str(proj)
    except Exception:
        pass

    gcloud_default = Path.home() / ".config" / "gcloud" / "configurations" / "config_default"
    if gcloud_default.is_file():
        try:
            for line in gcloud_default.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("project"):
                    parts = line.split("=", 1)
                    if len(parts) == 2 and parts[1].strip():
                        return parts[1].strip()
        except Exception:
            pass

    return None


def has_adc_credentials() -> bool:
    """Check if Google Cloud Application Default Credentials exist."""
    adc_env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc_env and Path(os.path.expandvars(os.path.expanduser(adc_env))).is_file():
        return True
    adc_default = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if adc_default.is_file():
        return True
    try:
        import google.auth
        creds, _ = google.auth.default()
        return creds is not None
    except Exception:
        return False


class IsolatedConfig(BaseModel):
    """Isolated configuration container.
    
    Fields are loaded strictly from the isolated env file or explicit CLI options,
    NEVER leaking into or falling back to global os.environ by default.
    """
    gemini_api_key: Optional[str] = Field(
        default=None,
        description="Gemini API Key for Google Antigravity SDK"
    )
    use_vertex: bool = Field(
        default=False,
        description="Whether to use Google Cloud Vertex AI via Application Default Credentials (ADC)"
    )
    gcp_project: Optional[str] = Field(
        default=None,
        description="Google Cloud Project ID for Vertex AI (auto-detected via ADC if omitted)"
    )
    gcp_location: str = Field(
        default="us-central1",
        description="Google Cloud Location/Region for Vertex AI"
    )
    google_application_credentials: Optional[str] = Field(
        default=None,
        description="Explicit path to Google Application Default Credentials JSON"
    )
    typesafe_api_key: Optional[str] = Field(
        default=None,
        description="TypeSafe API Key for Jev System One decision engine"
    )
    typesafe_base_url: str = Field(
        default="https://api.typesafe.ai",
        description="TypeSafe API base URL"
    )
    typesafe_model: str = Field(
        default="jev-latest",
        description="TypeSafe model identifier"
    )
    gemini_model: Optional[str] = Field(
        default=None,
        description="Google Gemini model identifier (default to SDK default if unset)"
    )
    max_context_tokens: int = Field(
        default=8000,
        description="Maximum tokens to allocate for generated context bundle"
    )
    relevance_threshold: float = Field(
        default=0.45,
        description="Jev Noul minimum probability for a file to be considered relevant"
    )
    isolate_env: bool = Field(
        default=True,
        description="Whether to isolate environment variables from global os.environ"
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: [
            ".git", ".venv", "venv", "node_modules", "__pycache__",
            ".pytest_cache", ".mypy_cache", ".DS_Store", "*.pyc",
            "*.lock", "package-lock.json", "dist", "build", "*.egg-info",
            ".idea", ".vscode", "*.min.js", "*.min.css", "*.map",
            "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.pdf",
            "*.zip", "*.tar", "*.gz", "*.mp3", "*.mp4", "*.dylib", "*.so"
        ],
        description="Default glob patterns to exclude during file crawling"
    )

    @classmethod
    def load(
        cls,
        env_file_path: Optional[str | Path] = None,
        allow_global_env: bool = False,
        overrides: Optional[Dict[str, Any]] = None
    ) -> IsolatedConfig:
        """Load isolated configuration.
        
        Priority:
        1. Explicit overrides passed as dict
        2. Specified env_file_path if provided
        3. Local ./.env.context_optimizer or ./.optimizer.env
        4. Global user config ~/.config/context-optimizer/env
        5. If allow_global_env is False, os.environ is NOT queried.
        """
        env_vars: Dict[str, str] = {}

        candidate_files: List[Path] = []
        if env_file_path:
            candidate_files.append(Path(os.path.expandvars(os.path.expanduser(str(env_file_path)))).resolve())
        else:
            candidate_files.append(LOCAL_ENV_FILE.resolve())
            candidate_files.append(OPTIMIZER_ENV_FILE.resolve())
            candidate_files.append(DEFAULT_ENV_FILE.expanduser().resolve())

        loaded_path: Optional[Path] = None
        for path in candidate_files:
            if path.is_file():
                loaded_path = path
                env_vars = cls._parse_env_file(path)
                break

        # Only check os.environ if allow_global_env is explicitly requested
        if allow_global_env:
            for key, val in os.environ.items():
                if key not in env_vars:
                    env_vars[key] = val

        # Map environment variable keys to model fields
        config_dict: Dict[str, Any] = {}
        if "GEMINI_API_KEY" in env_vars:
            config_dict["gemini_api_key"] = env_vars["GEMINI_API_KEY"]
        elif "CONTEXT_OPTIMIZER_GEMINI_KEY" in env_vars:
            config_dict["gemini_api_key"] = env_vars["CONTEXT_OPTIMIZER_GEMINI_KEY"]

        if "GOOGLE_APPLICATION_CREDENTIALS" in env_vars:
            cred_path = os.path.expandvars(os.path.expanduser(env_vars["GOOGLE_APPLICATION_CREDENTIALS"]))
            config_dict["google_application_credentials"] = cred_path
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

        if "CLOUDSDK_CONFIG" in env_vars:
            sdk_path = os.path.expandvars(os.path.expanduser(env_vars["CLOUDSDK_CONFIG"]))
            os.environ["CLOUDSDK_CONFIG"] = sdk_path

        use_vertex_env = (
            env_vars.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("true", "1", "yes") or
            env_vars.get("USE_VERTEX", "").lower() in ("true", "1", "yes") or
            env_vars.get("CONTEXT_OPTIMIZER_USE_VERTEX", "").lower() in ("true", "1", "yes") or
            env_vars.get("AGY_ADC_AUTH", "").lower() in ("true", "1", "yes")
        )

        project_val = (
            env_vars.get("GOOGLE_CLOUD_PROJECT") or
            env_vars.get("GCP_PROJECT") or
            env_vars.get("PROJECT_ID") or
            env_vars.get("CONTEXT_OPTIMIZER_GCP_PROJECT")
        )
        if project_val:
            config_dict["gcp_project"] = project_val

        location_val = (
            env_vars.get("GOOGLE_CLOUD_LOCATION") or
            env_vars.get("GCP_LOCATION") or
            env_vars.get("CONTEXT_OPTIMIZER_GCP_LOCATION")
        )
        if location_val:
            # Vertex AI regional publisher models require full region e.g. us-central1
            config_dict["gcp_location"] = "us-central1" if location_val.strip().lower() == "us" else location_val.strip()

        has_api_key = bool(config_dict.get("gemini_api_key"))
        adc_available = has_adc_credentials()

        if use_vertex_env or (not has_api_key and adc_available):
            config_dict["use_vertex"] = True
            if not config_dict.get("gcp_project"):
                detected_project = resolve_adc_project()
                if detected_project:
                    config_dict["gcp_project"] = detected_project

        if "TYPESAFE_API_KEY" in env_vars:
            config_dict["typesafe_api_key"] = env_vars["TYPESAFE_API_KEY"]
        elif "CONTEXT_OPTIMIZER_TYPESAFE_KEY" in env_vars:
            config_dict["typesafe_api_key"] = env_vars["CONTEXT_OPTIMIZER_TYPESAFE_KEY"]

        if "TYPESAFE_BASE_URL" in env_vars:
            config_dict["typesafe_base_url"] = env_vars["TYPESAFE_BASE_URL"]

        if "TYPESAFE_MODEL" in env_vars:
            model_val = env_vars["TYPESAFE_MODEL"].strip()
            config_dict["typesafe_model"] = "jev-latest" if model_val in {"jev", ""} else model_val

        model_name_env = env_vars.get("GEMINI_MODEL") or env_vars.get("OPTIMIZER_MODEL")
        if model_name_env:
            model_name_clean = model_name_env.strip()
            # If on Vertex, gemini-3.8-flash does not exist; normalize to gemini-2.5-flash
            if config_dict.get("use_vertex") and "gemini-3." in model_name_clean:
                config_dict["gemini_model"] = "gemini-2.5-flash"
            else:
                config_dict["gemini_model"] = model_name_clean

        if "MAX_CONTEXT_TOKENS" in env_vars:
            try:
                config_dict["max_context_tokens"] = int(env_vars["MAX_CONTEXT_TOKENS"])
            except ValueError:
                pass

        if "RELEVANCE_THRESHOLD" in env_vars:
            try:
                config_dict["relevance_threshold"] = float(env_vars["RELEVANCE_THRESHOLD"])
            except ValueError:
                pass

        if "EXCLUDE_PATTERNS" in env_vars:
            patterns = [p.strip() for p in env_vars["EXCLUDE_PATTERNS"].split(",") if p.strip()]
            if patterns:
                config_dict["exclude_patterns"] = patterns

        config_dict["isolate_env"] = not allow_global_env

        # Apply overrides
        if overrides:
            for k, v in overrides.items():
                if v is not None:
                    config_dict[k] = v

        return cls(**config_dict)

    @staticmethod
    def _parse_env_file(path: Path) -> Dict[str, str]:
        """Parse key=value pairs from a file without modifying os.environ."""
        result: Dict[str, str] = {}
        try:
            content = path.read_text(encoding="utf-8")
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    result[key] = os.path.expandvars(os.path.expanduser(val))
        except Exception:
            pass
        return result

    @classmethod
    def save_variable(cls, key: str, value: str, target_path: Optional[Path] = None) -> Path:
        """Persist a configuration variable to the isolated env file."""
        path = target_path or DEFAULT_ENV_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        current = cls._parse_env_file(path) if path.is_file() else {}
        current[key] = value

        lines = [f"# context-optimizer isolated environment configuration\n"]
        for k, v in sorted(current.items()):
            lines.append(f"{k}={v}\n")

        path.write_text("".join(lines), encoding="utf-8")
        path.chmod(0o600)  # secure permissions for credentials
        return path
