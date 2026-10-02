from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Provider = Literal["openai", "anthropic", "google", "gemini", "local", "minicpm"]
Tier = Literal["cheap", "standard", "frontier"]
Trigger = Literal["low_confidence", "high_risk_domain", "factual_claims", "code_generation"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=32000)


class Policy(StrictModel):
    privacy: Literal["public", "internal", "confidential", "restricted"] = "public"
    data_residency: str | None = Field(default=None, min_length=2, max_length=32)
    providers: list[Provider] = Field(
        default_factory=lambda: ["openai", "anthropic", "google", "local"], min_length=1
    )
    risk_tier: Literal["standard", "high"] = "standard"
    on_policy_violation: Literal["reject"] = "reject"


class Budget(StrictModel):
    max_total_cost_usd: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    max_latency_ms: int = Field(default=8000, gt=0, le=300000)
    max_wall_time_ms: int = Field(default=20000, gt=0, le=300000)
    max_model_calls: int = Field(default=4, ge=1, le=20)
    max_reasoning_tokens: int = Field(default=10000, ge=1, le=100000)
    on_exceeded: Literal["best_effort", "reject"] = "best_effort"


class Compute(StrictModel):
    strategy: Literal["adaptive", "fixed"] = "adaptive"
    initial_tier: Tier = "cheap"
    max_tier: Tier = "frontier"
    reasoning_effort: Literal["adaptive", "low", "medium", "high"] = "adaptive"
    escalate_on: list[Literal["complexity", "low_confidence", "verification_failure"]] = Field(
        default_factory=lambda: ["complexity", "low_confidence", "verification_failure"]
    )
    on_escalation_failure: Literal["best_available", "reject"] = "best_available"

    @model_validator(mode="after")
    def tier_order(self):
        if ["cheap", "standard", "frontier"].index(self.initial_tier) > [
            "cheap",
            "standard",
            "frontier",
        ].index(self.max_tier):
            raise ValueError("initial_tier must not exceed max_tier")
        return self


class Verification(StrictModel):
    mode: Literal["adaptive", "always", "off"] = "adaptive"
    triggers: list[Trigger] = Field(
        default_factory=lambda: [
            "low_confidence",
            "high_risk_domain",
            "factual_claims",
            "code_generation",
        ]
    )
    strategies: list[
        Literal["independent_model", "tool_check", "schema_validation", "consensus"]
    ] = Field(
        default_factory=lambda: [
            "independent_model",
            "tool_check",
            "schema_validation",
            "consensus",
        ]
    )
    failure_action: Literal["return_with_uncertainty", "reject"] = "return_with_uncertainty"


class ConsensusTrigger(StrictModel):
    confidence_below: float = Field(default=0.82, ge=0, le=1)


class Diversity(StrictModel):
    vendors: bool = True
    model_families: bool = True


class Consensus(StrictModel):
    trigger: ConsensusTrigger = Field(default_factory=ConsensusTrigger)
    min_independent_families: int = Field(default=3, ge=2, le=10)
    diversity: Diversity = Field(default_factory=Diversity)
    aggregation_strategy: Literal["@majority"] = "@majority"
    disagreement: Literal["return_with_uncertainty", "reject"] = "return_with_uncertainty"


class Memory(StrictModel):
    scope: Literal["user"] = "user"
    read: bool = False
    write: Literal["auto", "off"] = "off"
    retention: Literal["persistent", "request"] = "persistent"
    conflict_strategy: Literal["prefer_recent"] = "prefer_recent"


class Routing(StrictModel):
    strategy: Literal["adaptive", "fixed"] = "adaptive"
    optimize: list[Literal["quality", "cost", "latency"]] = Field(
        default_factory=lambda: ["quality", "cost", "latency"], min_length=1
    )
    optimization_mode: Literal["balanced"] = "balanced"
    fallback: Literal["next_best", "none"] = "next_best"
    provider_selection: Literal["policy_allowed"] = "policy_allowed"


class ResponseOptions(StrictModel):
    uncertainty: Literal["when_material", "always"] = "when_material"
    citations: Literal["auto", "off"] = "auto"
    format: Literal["auto", "text", "json"] = "auto"
    json_schema: dict[str, Any] | None = None

    @model_validator(mode="after")
    def check_schema(self):
        if self.json_schema is not None:
            import jsonschema

            try:
                jsonschema.Draft202012Validator.check_schema(self.json_schema)
            except jsonschema.SchemaError as exc:
                raise ValueError("Invalid response JSON Schema") from exc

            def has_ref(value):
                if isinstance(value, dict):
                    return (
                        "$ref" in value
                        or "$dynamicRef" in value
                        or any(has_ref(v) for v in value.values())
                    )
                return isinstance(value, list) and any(has_ref(v) for v in value)

            if has_ref(self.json_schema):
                raise ValueError("Schema references are not supported in this prototype")
            if self.format != "json":
                raise ValueError("json_schema requires response.format=json")
        return self


class ChatRequest(StrictModel):
    model: str = Field(default="@auto", min_length=1)
    provider: Provider | None = None
    user_id: str | None = Field(default=None, min_length=1, max_length=128)
    messages: list[Message] = Field(min_length=1, max_length=100)
    policy: Policy = Field(default_factory=Policy)
    budget: Budget = Field(default_factory=Budget)
    compute: Compute = Field(default_factory=Compute)
    verification: Verification = Field(default_factory=Verification)
    consensus: Consensus = Field(default_factory=Consensus)
    memory: Memory = Field(default_factory=Memory)
    routing: Routing = Field(default_factory=Routing)
    response: ResponseOptions = Field(default_factory=ResponseOptions)

    @model_validator(mode="after")
    def identity_and_size(self):
        if (self.memory.read or self.memory.write == "auto") and not self.user_id:
            raise ValueError("user_id is required when memory read or write is enabled")
        if sum(len(m.content.encode()) for m in self.messages) > 64000:
            raise ValueError("Combined messages exceed 64 KB")
        if not self.model.strip():
            raise ValueError("model must not be blank")
        return self


class Classification(StrictModel):
    privacy: Literal["public", "internal", "confidential", "restricted"]
    complexity: Literal["low", "medium", "high"]
    high_risk_domain: bool
    factual_claims: bool
    code_generation: bool


class Evaluation(StrictModel):
    confidence: float = Field(ge=0, le=1)
    passed: bool
    reason: str = Field(max_length=400)


class ChatResponse(StrictModel):
    provider: str
    model: str
    content: str
    finish_reason: str
    status: Literal["ok", "uncertain"]
    confidence: float | None
    uncertainty: list[str]
    citations: list[str] = Field(default_factory=list)
    classification: Classification | None
    usage: dict[str, Any]
    trace: list[dict[str, Any]]
