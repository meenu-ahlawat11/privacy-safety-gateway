"""Policy loading and action lookup."""

from pathlib import Path
from typing import Callable

import yaml

VALID_ACTIONS = ("allow", "mask", "redact", "tokenize", "block")


class PolicyError(Exception):
    """Raised when a policy file is missing or invalid."""


class Policy:
    """A validated policy: named profiles mapping detection types to actions."""

    def __init__(self, default_profile: str | None, profiles: dict[str, dict]) -> None:
        self._default_profile = default_profile
        self._profiles = profiles

    @classmethod
    def load(cls, path: str | Path = "policies/policy.yaml") -> "Policy":
        path = Path(path)
        if not path.is_file():
            raise PolicyError(f"policy file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PolicyError(f"invalid YAML in {path}: {exc}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
            raise PolicyError("policy file must contain a 'profiles' mapping")

        default_profile = data.get("default_profile")
        profiles: dict[str, dict] = {}
        for name, profile in data["profiles"].items():
            if not isinstance(profile, dict):
                raise PolicyError(f"profile {name!r} must be a mapping")
            default_action = profile.get("default_action")
            if default_action is None:
                raise PolicyError(f"profile {name!r} is missing 'default_action'")
            if default_action not in VALID_ACTIONS:
                raise PolicyError(
                    f"profile {name!r} has invalid default_action {default_action!r}"
                )
            actions = profile.get("actions", {})
            if not isinstance(actions, dict):
                raise PolicyError(f"profile {name!r}: 'actions' must be a mapping")
            for det_type, action in actions.items():
                if action not in VALID_ACTIONS:
                    raise PolicyError(
                        f"profile {name!r}: invalid action {action!r} for {det_type!r}"
                    )
            profiles[str(name)] = {
                "default_action": default_action,
                "actions": {str(k): v for k, v in actions.items()},
            }

        if default_profile is not None and default_profile not in profiles:
            raise PolicyError(f"default_profile {default_profile!r} does not exist")

        return cls(default_profile=default_profile, profiles=profiles)

    def profile_names(self) -> list[str]:
        return list(self._profiles)

    def action_for(self, profile_name: str | None = None) -> Callable[[str], str]:
        """Return a function mapping detection type -> action for a profile."""
        if profile_name is None:
            if self._default_profile is None:
                raise PolicyError("no profile specified and no default_profile set")
            profile_name = self._default_profile
        if profile_name not in self._profiles:
            raise PolicyError(f"unknown profile: {profile_name!r}")
        profile = self._profiles[profile_name]
        actions: dict[str, str] = profile["actions"]
        default_action: str = profile["default_action"]

        def lookup(detection_type: str) -> str:
            return actions.get(detection_type, default_action)

        return lookup
