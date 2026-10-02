from dataclasses import dataclass
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

from app.schemas import Tier

TIERS = {"cheap": 0, "standard": 1, "frontier": 2}
PRIVACY = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


class Metadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    family: str
    vendor: str
    tier: Tier
    region: str | None = None
    confidential_approved: bool = False
    input_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    output_usd_per_million: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    expected_latency_ms: int = Field(default=3000, gt=0)
    quality: float = Field(default=0.5, ge=0, le=1)


@dataclass(frozen=True)
class Deployment:
    provider: str
    metadata: Metadata
    is_local: bool

    @property
    def model(self):
        return self.metadata.model

    @property
    def tier(self):
        return TIERS[self.metadata.tier]


def canonical(provider):
    return "gemini" if provider == "google" else provider


def local_url(url):
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


def catalog(settings, configs, clients):
    defaults = {
        "minicpm": ("openbmb-minicpm5", "openbmb", "cheap", 0.4, 700),
        "local": ("qwen3-vl", "alibaba", "standard", 0.65, 1500),
        "openai": ("gpt", "openai", "frontier", 0.9, 3000),
        "gemini": ("gemini", "google", "frontier", 0.9, 3000),
        "anthropic": ("claude", "anthropic", "frontier", 0.9, 3000),
    }
    result = []
    for provider, (url, _, model) in configs.items():
        if provider not in clients or not model:
            continue
        is_local = provider in {"local", "minicpm"} and local_url(url)
        family, vendor, tier, quality, latency = defaults[provider]
        metadata = Metadata(
            model=model,
            family=family,
            vendor=vendor,
            tier=tier,
            region=settings.local_data_residency if is_local else None,
            confidential_approved=is_local,
            input_usd_per_million=0 if is_local else None,
            output_usd_per_million=0 if is_local else None,
            quality=quality,
            expected_latency_ms=latency,
        )
        override = settings.model_metadata.get(
            provider, settings.model_metadata.get("google", {}) if provider == "gemini" else {}
        )
        if override:
            if override.get("model") != model:
                raise ValueError(
                    f"MODEL_METADATA for {provider} must name its configured model exactly"
                )
            metadata = Metadata.model_validate({**metadata.model_dump(), **override})
        result.append(Deployment(provider, metadata, is_local))
    return result


def allowed(deployment, policy, privacy):
    providers = {canonical(p) for p in policy.providers}
    name = "local" if deployment.is_local else deployment.provider
    if name not in providers and deployment.provider not in providers:
        return False
    if (
        policy.data_residency
        and (deployment.metadata.region or "").upper() != policy.data_residency.upper()
    ):
        return False
    if privacy == "restricted" and not deployment.is_local:
        return False
    return not (
        PRIVACY[privacy] >= PRIVACY["confidential"]
        and not deployment.metadata.confidential_approved
    )


def select_candidates(deployments, request, privacy, target):
    candidates = [
        d
        for d in deployments
        if allowed(d, request.policy, privacy) and d.tier <= TIERS[request.compute.max_tier]
    ]
    if request.provider:
        candidates = [d for d in candidates if d.provider == canonical(request.provider)]
    if request.model != "@auto":
        # Registered models only: otherwise price/residency/approval metadata would be wrong.
        candidates = [d for d in candidates if d.model == request.model]
    if request.compute.strategy == "fixed" or request.routing.strategy == "fixed":
        candidates = [d for d in candidates if d.tier == target]

    def score(d):
        costs = [
            x.metadata.output_usd_per_million
            for x in candidates
            if x.metadata.output_usd_per_million is not None
        ]
        max_cost = max(costs, default=1) or 1
        metrics = {
            "quality": d.metadata.quality,
            "cost": 1
            - min(
                (
                    d.metadata.output_usd_per_million
                    if d.metadata.output_usd_per_million is not None
                    else max_cost
                )
                / max_cost,
                1,
            ),
            "latency": 1 / (1 + d.metadata.expected_latency_ms / 1000),
        }
        return sum(metrics[k] for k in set(request.routing.optimize)) / len(
            set(request.routing.optimize)
        )

    return sorted(candidates, key=lambda d: (abs(d.tier - target), -score(d), d.provider))
