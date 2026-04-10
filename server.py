import asyncio
import base64
import json
import os
import re
import secrets
import signal
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from starlette.applications import Starlette
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    SimpleUser,
)
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route
from starlette.templating import Jinja2Templates

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = secrets.token_urlsafe(16)
    print(f"Generated admin password: {ADMIN_PASSWORD}")

HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
ENV_FILE_PATH = Path(HERMES_HOME) / ".env"
PAIRING_DIR = Path(HERMES_HOME) / "pairing"
CODE_TTL_SECONDS = 3600

ENV_VAR_DEFS = [
    # Model
    ("LLM_MODEL", "Model", "model", False),
    # Providers
    ("OPENROUTER_API_KEY", "OpenRouter API Key", "provider", True),
    ("DEEPSEEK_API_KEY", "DeepSeek API Key", "provider", True),
    ("DASHSCOPE_API_KEY", "DashScope API Key", "provider", True),
    ("GLM_API_KEY", "GLM / Z.AI API Key", "provider", True),
    ("KIMI_API_KEY", "Kimi API Key", "provider", True),
    ("MINIMAX_API_KEY", "MiniMax API Key", "provider", True),
    ("HF_TOKEN", "Hugging Face Token", "provider", True),
    # Tools
    ("PARALLEL_API_KEY", "Parallel API Key", "tool", True),
    ("FIRECRAWL_API_KEY", "Firecrawl API Key", "tool", True),
    ("TAVILY_API_KEY", "Tavily API Key", "tool", True),
    ("FAL_KEY", "FAL API Key", "tool", True),
    ("BROWSERBASE_API_KEY", "Browserbase API Key", "tool", True),
    ("BROWSERBASE_PROJECT_ID", "Browserbase Project ID", "tool", False),
    ("GITHUB_TOKEN", "GitHub Token", "tool", True),
    ("VOICE_TOOLS_OPENAI_KEY", "OpenAI Voice Key", "tool", True),
    ("HONCHO_API_KEY", "Honcho API Key", "tool", True),
    # Messaging — Telegram
    ("TELEGRAM_BOT_TOKEN", "Telegram Bot Token", "messaging", True),
    ("TELEGRAM_ALLOWED_USERS", "Telegram Allowed Users", "messaging", False),
    # Messaging — Discord
    ("DISCORD_BOT_TOKEN", "Discord Bot Token", "messaging", True),
    ("DISCORD_ALLOWED_USERS", "Discord Allowed Users", "messaging", False),
    # Messaging — Slack
    ("SLACK_BOT_TOKEN", "Slack Bot Token", "messaging", True),
    ("SLACK_APP_TOKEN", "Slack App Token", "messaging", True),
    # Messaging — WhatsApp
    ("WHATSAPP_ENABLED", "WhatsApp Enabled", "messaging", False),
    # Messaging — Email
    ("EMAIL_ADDRESS", "Email Address", "messaging", False),
    ("EMAIL_PASSWORD", "Email Password", "messaging", True),
    ("EMAIL_IMAP_HOST", "Email IMAP Host", "messaging", False),
    ("EMAIL_SMTP_HOST", "Email SMTP Host", "messaging", False),
    # Messaging — Mattermost
    ("MATTERMOST_URL", "Mattermost URL", "messaging", False),
    ("MATTERMOST_TOKEN", "Mattermost Token", "messaging", True),
    # Messaging — Matrix
    ("MATRIX_HOMESERVER", "Matrix Homeserver", "messaging", False),
    ("MATRIX_ACCESS_TOKEN", "Matrix Access Token", "messaging", True),
    ("MATRIX_USER_ID", "Matrix User ID", "messaging", False),
    # Messaging — General
    ("GATEWAY_ALLOW_ALL_USERS", "Allow All Users", "messaging", False),
]

PASSWORD_KEYS = {key for key, _, _, is_pw in ENV_VAR_DEFS if is_pw}

PROVIDER_KEYS = [key for key, _, cat, _ in ENV_VAR_DEFS if cat == "provider" and key != "LLM_MODEL"]
CHANNEL_KEYS = {
    "Telegram": "TELEGRAM_BOT_TOKEN",
    "Discord": "DISCORD_BOT_TOKEN",
    "Slack": "SLACK_BOT_TOKEN",
    "WhatsApp": "WHATSAPP_ENABLED",
    "Email": "EMAIL_ADDRESS",
    "Mattermost": "MATTERMOST_TOKEN",
    "Matrix": "MATRIX_ACCESS_TOKEN",
}


def read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        result[key] = value
    return result


def write_env_file(path: Path, env_vars: dict[str, str]):
    path.parent.mkdir(parents=True, exist_ok=True)

    categories = {"model": "Model", "provider": "Providers", "tool": "Tools", "messaging": "Messaging"}
    grouped: dict[str, list[str]] = {cat: [] for cat in categories}
    key_to_cat = {key: cat for key, _, cat, _ in ENV_VAR_DEFS}

    for key, value in env_vars.items():
        if not value:
            continue
        cat = key_to_cat.get(key, "other")
        line = f"{key}={value}"
        if cat in grouped:
            grouped[cat].append(line)
        else:
            grouped.setdefault("other", []).append(line)

    lines = []
    for cat, heading in categories.items():
        entries = grouped.get(cat, [])
        if entries:
            lines.append(f"# {heading}")
            lines.extend(sorted(entries))
            lines.append("")

    other = grouped.get("other", [])
    if other:
        lines.append("# Other")
        lines.extend(sorted(other))
        lines.append("")

    path.write_text("\n".join(lines) + "\n" if lines else "")


def mask_secrets(env_vars: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, value in env_vars.items():
        if key in PASSWORD_KEYS and value:
            result[key] = value[:8] + "***" if len(value) > 8 else "***"
        else:
            result[key] = value
    return result


def merge_secrets(new_vars: dict[str, str], existing_vars: dict[str, str]) -> dict[str, str]:
    result = {}
    for key, value in new_vars.items():
        if key in PASSWORD_KEYS and value.endswith("***"):
            result[key] = existing_vars.get(key, "")
        else:
            result[key] = value
    return result


class BasicAuthBackend(AuthenticationBackend):
    async def authenticate(self, conn):
        if "Authorization" not in conn.headers:
            return None

        auth = conn.headers["Authorization"]
        try:
            scheme, credentials = auth.split()
            if scheme.lower() != "basic":
                return None
            decoded = base64.b64decode(credentials).decode("ascii")
        except (ValueError, UnicodeDecodeError):
            raise AuthenticationError("Invalid credentials")

        username, _, password = decoded.partition(":")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            return AuthCredentials(["authenticated"]), SimpleUser(username)

        raise AuthenticationError("Invalid credentials")


def require_auth(request: Request):
    if not request.user.is_authenticated:
        return PlainTextResponse(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="hermes"'},
        )
    return None


class GatewayManager:
    def __init__(self):
        self.process: asyncio.subprocess.Process | None = None
        self.state = "stopped"
        self.logs: deque[str] = deque(maxlen=500)
        self.start_time: float | None = None
        self.restart_count = 0
        self._read_tasks: list[asyncio.Task] = []

    async def start(self):
        if self.process and self.process.returncode is None:
            return
        self.state = "starting"
        try:
            env = os.environ.copy()
            env["HERMES_HOME"] = HERMES_HOME
            env_vars = read_env_file(ENV_FILE_PATH)
            env.update(env_vars)

            llm_model = env_vars.get("LLM_MODEL") or os.environ.get("LLM_MODEL")
            if llm_model:
                import yaml as _yaml
                _cp = Path(HERMES_HOME) / "config.yaml"
                try:
                    _cfg = _yaml.safe_load(_cp.read_text()) if _cp.exists() else {}
                    _cfg = _cfg or {}
                    _cfg.setdefault("model", {})
                    _cfg["model"]["default"] = llm_model
                    _cp.write_text(_yaml.dump(_cfg))
                except Exception as _e:
                    self.logs.append(f"Model patch: {_e}")

            self.process = await asyncio.create_subprocess_exec(
                "hermes", "gateway",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=env,
            )
            self.state = "running"
            self.start_time = time.time()
            task = asyncio.create_task(self._read_output())
            self._read_tasks.append(task)
        except Exception as e:
            self.state = "error"
            self.logs.append(f"Failed to start gateway: {e}")

    async def stop(self):
        if not self.process or self.process.returncode is not None:
            self.state = "stopped"
            return
        self.state = "stopping"
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()
        self.state = "stopped"
        self.start_time = None

    async def restart(self):
        await self.stop()
        self.restart_count += 1
        await self.start()

    async def _read_output(self):
        try:
            while self.process and self.process.stdout:
                line = await self.process.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                cleaned = ANSI_ESCAPE.sub("", decoded)
                self.logs.append(cleaned)
        except asyncio.CancelledError:
            return
        if self.process and self.process.returncode is not None and self.state == "running":
            self.state = "error"
            self.logs.append(f"Gateway exited with code {self.process.returncode}")

    def get_status(self) -> dict:
        pid = None
        if self.process and self.process.returncode is None:
            pid = self.process.pid
        uptime = None
        if self.start_time and self.state == "running":
            uptime = int(time.time() - self.start_time)
        return {
            "state": self.state,
            "pid": pid,
            "uptime": uptime,
            "restart_count": self.restart_count,
        }


gateway = GatewayManager()
config_lock = asyncio.Lock()


async def homepage(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    return templates.TemplateResponse(request, "index.html")


async def health(request: Request):
    return JSONResponse({"status": "ok", "gateway": gateway.state})


async def api_config_get(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err
    async with config_lock:
        env_vars = read_env_file(ENV_FILE_PATH)
    defs = [
        {"key": key, "label": label, "category": cat, "password": is_pw}
        for key, label, cat, is_pw in ENV_VAR_DEFS
    ]
    return JSONResponse({"vars": mask_secrets(env_vars), "defs": defs})


async def api_config_put(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    try:
        restart = body.pop("_restartGateway", False)
        new_vars = body.get("vars", {})

        async with config_lock:
            existing = read_env_file(ENV_FILE_PATH)
            merged = merge_secrets(new_vars, existing)
            for key, value in existing.items():
                if key not in merged:
                    merged[key] = value
            write_env_file(ENV_FILE_PATH, merged)

        if restart:
            asyncio.create_task(gateway.restart())

        return JSONResponse({"ok": True, "restarting": restart})
    except Exception as e:
        print(f"Config save error: {type(e).__name__}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


async def api_status(request: Request):
    auth_err = require_auth(request)
    if auth_err:
        return auth_err

    env_vars = read_env_file(ENV_FILE_PATH)

    providers = {}
    for key in PROVIDER_KEYS:
        label = key.replace("_API_KEY", "").replace("_TOKEN", "").replace("HF_", "HuggingFace ").replace("_", " ").title()
        providers
