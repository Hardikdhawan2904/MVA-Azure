from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class KPIDefinition(BaseModel):
    label: str
    abbreviation: Optional[str] = None
    actual_column: Optional[str] = None
    budget_column: Optional[str] = None
    prior_year_column: Optional[str] = None
    formula: Optional[str] = None
    unit: str
    higher_is_better: bool
    category: str
    dependencies: List[str] = Field(default_factory=list)

    model_config = {
        "extra": "allow"
    }

class KPIDefinitionsRoot(BaseModel):
    kpis: Dict[str, KPIDefinition]

class BusinessRule(BaseModel):
    id: str
    name: str
    category: str
    kpi: str
    condition: str
    interpretation: str
    severity: str
    action: str

    model_config = {
        "extra": "allow"
    }

class BusinessRulesRoot(BaseModel):
    predefined_rules: List[BusinessRule] = Field(default_factory=list)
    thresholds: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    variance_drivers: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    flags: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    new_rules: List[BusinessRule] = Field(default_factory=list)
    supported_operations: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = {
        "extra": "allow"
    }
