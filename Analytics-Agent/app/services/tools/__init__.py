"""app/services/tools/ — the 6 concrete tools the analyzers/pipeline call
into to do real work (SQL retrieval, math, ML models, LLM narration,
curated-KPI lookup), plus KnowledgeUpdateTool, a closely related
supporting service. Grouped here for the same reason every other Stage
1-9 concern (analyzers/, planning/, kpi_discovery/, ...) lives in its own
subdirectory rather than app/services/ directly — each file has a
distinct concern and its own heavy dependency (duckdb, scikit-learn/
prophet/lightgbm, the Azure OpenAI client), so keeping them separate avoids
transitively importing everything just to use one.
"""
