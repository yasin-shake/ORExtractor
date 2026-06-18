"""Pydantic models describing the structured data extracted from NI 43-101 reports.

NI 43-101 ("Standards of Disclosure for Mineral Projects") is the Canadian
technical reporting standard for mineral projects. These models capture the
fields that are most commonly summarised in such reports. Every field is
optional because any individual report may omit a given section, and the
extractor must degrade gracefully (returning ``None``) when data is absent
rather than fabricating values.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, Field, field_validator


def _coerce_to_list(v: Any) -> list:
    """Coerce LLM list-field responses that arrive as bare strings or JSON 'null'."""
    if v is None:
        return []
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() == "null":
            return []
        return [s]
    return v


class GradeEntry(BaseModel):
    """One commodity's grade and contained metal within a resource/reserve row.

    Used in polymetallic deposits where a single row reports Cu%, Au g/t and
    Ag g/t simultaneously.  For single-commodity deposits the list will contain
    exactly one entry.
    """

    commodity: Optional[str] = Field(
        None, description="Commodity symbol or name (e.g. 'Au', 'Cu', 'Ag', 'Mo')."
    )
    grade_value: Optional[float] = Field(None, description="Numeric grade value.")
    grade_unit: Optional[str] = Field(
        None, description="Grade unit (e.g. 'g/t', '%', 'ppm', 'ppb')."
    )
    contained_metal: Optional[float] = Field(None, description="Contained metal quantity.")
    contained_metal_unit: Optional[str] = Field(
        None, description="Unit of contained metal (e.g. 'koz', 'Moz', 'kt', 'Mlb')."
    )


class PropertyInfo(BaseModel):
    """High-level information about the property / project."""

    project_name: Optional[str] = Field(
        None, description="Name of the mineral project or property."
    )
    country: Optional[str] = Field(None, description="Country where the property is located.")
    region: Optional[str] = Field(
        None, description="State, province, district or region of the property."
    )
    coordinates: Optional[str] = Field(
        None,
        description="Geographic coordinates (latitude/longitude or UTM) as stated in the report.",
    )
    latitude: Optional[float] = Field(
        None,
        description=(
            "Decimal-degree latitude of the property centroid "
            "(positive = North, negative = South). Convert DMS to decimal if needed."
        ),
    )
    longitude: Optional[float] = Field(
        None,
        description=(
            "Decimal-degree longitude of the property centroid "
            "(positive = East, negative = West). Convert DMS to decimal if needed."
        ),
    )
    jurisdiction: Optional[str] = Field(
        None,
        description=(
            "Mining jurisdiction — typically 'Province/State, Country' "
            "(e.g. 'Ontario, Canada', 'Nevada, USA', 'Western Australia, Australia')."
        ),
    )
    exchange_listed: Optional[str] = Field(
        None,
        description=(
            "Stock exchange(s) where the issuer company is listed "
            "(e.g. 'TSX', 'TSX-V', 'ASX', 'NYSE', 'OTCQB'). "
            "Look for exchange ticker symbols in the report header or cover page."
        ),
    )
    project_stage: Optional[str] = Field(
        None,
        description=(
            "Project development stage as described in the report. "
            "Use one of: Grassroots, Exploration, Resource Definition, "
            "PEA, Pre-Feasibility, Feasibility, Permitted, Construction, "
            "Operating, Care & Maintenance, Closed."
        ),
    )
    area_hectares: Optional[float] = Field(
        None, description="Total property area in hectares, if reported."
    )
    ownership: Optional[str] = Field(
        None,
        description="Ownership structure, holders and percentage interests in the property.",
    )
    commodities: List[str] = Field(
        default_factory=list,
        description="Primary commodities of interest (e.g. gold, copper, lithium).",
    )

    @field_validator("commodities", mode="before")
    @classmethod
    def _coerce_commodities(cls, v: Any) -> list:
        return _coerce_to_list(v)

    tenure_status: Optional[str] = Field(
        None,
        description="Mineral tenure / claim / licence status and key permitting notes.",
    )
    accessibility: Optional[str] = Field(
        None,
        description="How the property is accessed (road, air, season, distance from nearest town).",
    )
    infrastructure: Optional[str] = Field(
        None,
        description="Available or planned infrastructure: power, water, roads, port, camp.",
    )


class MineralResource(BaseModel):
    """A single line item from a mineral resource estimate table."""

    category: Optional[str] = Field(
        None, description="Resource category: Measured, Indicated or Inferred."
    )
    commodity: Optional[str] = Field(
        None, description="Primary commodity this row refers to (e.g. 'Au', 'Cu')."
    )
    zone: Optional[str] = Field(
        None, description="Deposit, zone or domain this line item refers to, if specified."
    )
    cut_off_grade: Optional[str] = Field(
        None, description="Cut-off grade applied, including units (e.g. '0.5 g/t Au')."
    )
    tonnes: Optional[float] = Field(
        None, description="Tonnage for this category (in tonnes as reported)."
    )
    grades: List[GradeEntry] = Field(
        default_factory=list,
        description=(
            "Grade and contained metal for every commodity in this row. "
            "Polymetallic rows will have multiple entries (e.g. Cu, Au, Ag). "
            "Single-commodity rows will have one entry."
        ),
    )

    @field_validator("grades", mode="before")
    @classmethod
    def _coerce_grades(cls, v: Any) -> list:
        return _coerce_to_list(v)

    effective_date: Optional[str] = Field(
        None, description="Effective date of the resource estimate."
    )


class MineralReserve(BaseModel):
    """A single line item from a mineral reserve estimate table."""

    category: Optional[str] = Field(
        None, description="Reserve category: Proven or Probable."
    )
    commodity: Optional[str] = Field(
        None, description="Primary commodity this row refers to (e.g. 'Au', 'Cu')."
    )
    zone: Optional[str] = Field(
        None, description="Deposit, zone or domain this line item refers to, if specified."
    )
    cut_off_grade: Optional[str] = Field(
        None, description="Cut-off grade applied, including units."
    )
    tonnes: Optional[float] = Field(
        None, description="Tonnage for this category (in tonnes as reported)."
    )
    grades: List[GradeEntry] = Field(
        default_factory=list,
        description=(
            "Grade and contained metal for every commodity in this row. "
            "Polymetallic rows will have multiple entries. "
            "Single-commodity rows will have one entry."
        ),
    )

    @field_validator("grades", mode="before")
    @classmethod
    def _coerce_grades(cls, v: Any) -> list:
        return _coerce_to_list(v)

    effective_date: Optional[str] = Field(
        None, description="Effective date of the reserve estimate."
    )


class EconomicParameters(BaseModel):
    """Project economics, typically from a PEA, PFS or feasibility study."""

    study_type: Optional[str] = Field(
        None,
        description="Type of economic study (PEA, Pre-Feasibility, Feasibility).",
    )
    study_effective_date: Optional[str] = Field(
        None, description="Effective date of the economic study."
    )
    pre_tax_npv: Optional[str] = Field(
        None, description="Pre-tax net present value, including currency and units."
    )
    post_tax_npv: Optional[str] = Field(
        None, description="Post-tax net present value, including currency and units."
    )
    irr: Optional[str] = Field(
        None, description="Internal rate of return (pre- and/or post-tax) as reported."
    )
    payback_years: Optional[float] = Field(
        None, description="Payback period in years."
    )
    discount_rate: Optional[str] = Field(
        None, description="Discount rate used in the NPV calculation (e.g. '5%')."
    )
    initial_capex: Optional[str] = Field(
        None, description="Initial capital expenditure, including currency and units."
    )
    sustaining_capex: Optional[str] = Field(
        None, description="Sustaining capital expenditure over the mine life."
    )
    total_capex: Optional[str] = Field(
        None, description="Total capital expenditure (initial + sustaining), if reported."
    )
    opex: Optional[str] = Field(
        None,
        description="Operating cost per tonne milled or per unit metal (AISC).",
    )
    mine_life_years: Optional[float] = Field(
        None, description="Projected life of mine in years."
    )
    throughput_tpd: Optional[float] = Field(
        None, description="Processing plant throughput in tonnes per day."
    )
    strip_ratio: Optional[str] = Field(
        None, description="Waste-to-ore strip ratio for open-pit operations (e.g. '3.2:1')."
    )
    recovery_rate: Optional[str] = Field(
        None,
        description=(
            "Metallurgical recovery rate(s) as reported "
            "(e.g. '92% Cu, 88% Au' or '94%'). Preserve all commodities."
        ),
    )
    royalties: Optional[str] = Field(
        None, description="Royalty obligations (NSR, NPI, gross revenue) and rates."
    )
    metal_price_assumptions: List[str] = Field(
        default_factory=list,
        description="Commodity price assumptions used in the economic model.",
    )

    @field_validator("metal_price_assumptions", mode="before")
    @classmethod
    def _coerce_metal_price_assumptions(cls, v: Any) -> list:
        return _coerce_to_list(v)


class GeologySummary(BaseModel):
    """Summary of the geological setting and mineralisation."""

    deposit_type: Optional[str] = Field(
        None, description="Classification of the deposit (e.g. epithermal, porphyry, VMS, IOCG)."
    )
    geological_age: Optional[str] = Field(
        None, description="Geological age of the host rocks or mineralising event."
    )
    host_rock: Optional[str] = Field(None, description="Dominant host rock lithologies.")
    mineralization_style: Optional[str] = Field(
        None, description="Style and form of mineralisation."
    )
    structural_controls: Optional[str] = Field(
        None, description="Key structural controls on mineralisation."
    )
    alteration: Optional[str] = Field(
        None, description="Notable alteration assemblages associated with mineralisation."
    )
    historical_production: Optional[str] = Field(
        None, description="Any recorded historical mining or production on or near the property."
    )


class ExplorationSummary(BaseModel):
    """Summary of exploration work completed on the property."""

    total_drill_holes: Optional[int] = Field(
        None, description="Total number of drill holes reported."
    )
    total_metres_drilled: Optional[float] = Field(
        None, description="Total metres drilled across all programs."
    )
    drilling_types: List[str] = Field(
        default_factory=list,
        description="Types of drilling performed (e.g. diamond, RC, auger).",
    )

    @field_validator("drilling_types", "notable_intercepts", "geophysical_surveys", mode="before")
    @classmethod
    def _coerce_str_lists(cls, v: Any) -> list:
        return _coerce_to_list(v)

    last_program_date: Optional[str] = Field(
        None, description="Date or year of the most recent exploration/drill program."
    )
    sampling_methods: Optional[str] = Field(
        None, description="Sampling and assay methods used (QA/QC summary)."
    )
    notable_intercepts: List[str] = Field(
        default_factory=list,
        description="Highlighted significant drill intercepts, if any.",
    )
    geophysical_surveys: List[str] = Field(
        default_factory=list,
        description="Geophysical survey types completed (e.g. IP, CSAMT, airborne magnetics).",
    )


class MiningMethod(BaseModel):
    """Mining method and key mine-design parameters."""

    method: Optional[str] = Field(
        None,
        description="Primary mining method (e.g. 'Open Pit', 'Underground', 'Combined open pit and underground').",
    )
    mining_rate_tpd: Optional[float] = Field(
        None, description="Designed ore mining rate in tonnes per day."
    )
    strip_ratio: Optional[str] = Field(
        None, description="Waste-to-ore strip ratio for open-pit (e.g. '3.2:1 waste:ore')."
    )
    dilution: Optional[str] = Field(
        None, description="Planned dilution percentage or tonnes added."
    )
    mine_recovery: Optional[str] = Field(
        None, description="Planned mine recovery factor (ore extracted vs in-situ)."
    )
    key_equipment: List[str] = Field(
        default_factory=list,
        description="Key mining equipment or fleet listed in the report.",
    )

    @field_validator("key_equipment", mode="before")
    @classmethod
    def _coerce_key_equipment(cls, v: Any) -> list:
        return _coerce_to_list(v)


class ProcessingMethod(BaseModel):
    """Processing / metallurgical method and plant parameters."""

    method: Optional[str] = Field(
        None,
        description="Processing method (e.g. 'Conventional flotation', 'CIL', 'Heap leach', 'SART').",
    )
    throughput_tpd: Optional[float] = Field(
        None, description="Design plant throughput in tonnes per day."
    )
    recoveries: List[str] = Field(
        default_factory=list,
        description="Metallurgical recovery by commodity (e.g. 'Cu: 92%', 'Au: 88%', 'Ag: 75%').",
    )

    @field_validator("recoveries", mode="before")
    @classmethod
    def _coerce_recoveries(cls, v: Any) -> list:
        return _coerce_to_list(v)

    concentrate_grade: Optional[str] = Field(
        None, description="Target concentrate grade specification, if applicable."
    )
    tailings_management: Optional[str] = Field(
        None, description="Tailings storage and management approach."
    )


class EnvironmentalSummary(BaseModel):
    """Environmental and permitting status."""

    permit_status: Optional[str] = Field(
        None, description="Current permitting status and summary of key permits received."
    )
    key_permits_required: List[str] = Field(
        default_factory=list,
        description="Major permits still required before construction or operation.",
    )
    environmental_studies_completed: List[str] = Field(
        default_factory=list,
        description="Completed baseline or environmental impact studies.",
    )
    tailings_facility: Optional[str] = Field(
        None, description="Tailings storage facility design type and planned location."
    )
    water_management: Optional[str] = Field(
        None, description="Water management strategy (source, treatment, discharge)."
    )
    closure_cost: Optional[str] = Field(
        None, description="Estimated reclamation and closure cost, including currency."
    )
    indigenous_consultation: Optional[str] = Field(
        None,
        description=(
            "Indigenous and community consultation status: duty-to-consult obligations, "
            "impact benefit agreements (IBAs), free prior and informed consent (FPIC) "
            "status, and any community opposition noted."
        ),
    )
    political_risk_flags: List[str] = Field(
        default_factory=list,
        description=(
            "Political, sovereign, or social risk flags noted in the report "
            "(e.g. 'Resource nationalism risk', 'Mining code under review', "
            "'Community blockades noted', 'Operating in conflict-affected region')."
        ),
    )

    @field_validator(
        "key_permits_required", "environmental_studies_completed", "political_risk_flags",
        mode="before",
    )
    @classmethod
    def _coerce_env_lists(cls, v: Any) -> list:
        return _coerce_to_list(v)


class QualifiedPerson(BaseModel):
    """A Qualified Person (QP) responsible for part or all of the report."""

    name: Optional[str] = Field(None, description="Full name of the qualified person.")
    credentials: Optional[str] = Field(
        None, description="Professional designation (e.g. P.Geo, P.Eng) and affiliation."
    )
    responsibility: Optional[str] = Field(
        None, description="Sections of the report this QP is responsible for."
    )


class NI43101Report(BaseModel):
    """Top-level structured representation of an NI 43-101 technical report."""

    source_file: Optional[str] = Field(
        None, description="Filename of the source PDF this data was extracted from."
    )
    report_title: Optional[str] = Field(None, description="Full title of the technical report.")
    report_date: Optional[str] = Field(
        None, description="Report date or effective date of the technical report."
    )
    report_purpose: Optional[str] = Field(
        None,
        description=(
            "Purpose / trigger for the report "
            "(e.g. 'Initial NI 43-101 technical report', "
            "'Updated resource estimate', 'Preliminary Economic Assessment', "
            "'Pre-Feasibility Study', 'Feasibility Study', "
            "'Material Change — updated mineral reserves')."
        ),
    )
    previous_resource_date: Optional[str] = Field(
        None,
        description=(
            "Effective date of the prior resource or reserve estimate that this report "
            "supersedes or updates, if stated (e.g. 'June 15, 2021'). "
            "Leave null if this is the first estimate for the project."
        ),
    )
    issuer: Optional[str] = Field(
        None, description="Issuer / company on whose behalf the report was prepared."
    )
    authors: List[str] = Field(
        default_factory=list,
        description="Authoring firms or individuals credited on the report.",
    )
    qualified_persons: List[QualifiedPerson] = Field(
        default_factory=list,
        description="Qualified Persons responsible for the report.",
    )

    @field_validator("authors", "qualified_persons", mode="before")
    @classmethod
    def _coerce_author_lists(cls, v: Any) -> list:
        return _coerce_to_list(v)
    property_info: Optional[PropertyInfo] = Field(
        None, description="Property / project location and ownership information."
    )
    geology: Optional[GeologySummary] = Field(
        None, description="Geological setting and mineralisation summary."
    )
    exploration: Optional[ExplorationSummary] = Field(
        None, description="Summary of exploration and drilling work."
    )
    mineral_resources: List[MineralResource] = Field(
        default_factory=list,
        description="Line items from the mineral resource estimate.",
    )
    mineral_reserves: List[MineralReserve] = Field(
        default_factory=list,
        description="Line items from the mineral reserve estimate.",
    )

    @field_validator("mineral_resources", "mineral_reserves", mode="before")
    @classmethod
    def _coerce_mineral_lists(cls, v: Any) -> list:
        return _coerce_to_list(v)
    economics: Optional[EconomicParameters] = Field(
        None, description="Project economic parameters."
    )
    mining_method: Optional[MiningMethod] = Field(
        None, description="Mining method and mine-design parameters."
    )
    processing_method: Optional[ProcessingMethod] = Field(
        None, description="Processing / metallurgical method and plant parameters."
    )
    environmental: Optional[EnvironmentalSummary] = Field(
        None, description="Environmental baseline and permitting status."
    )
    summary: Optional[str] = Field(
        None,
        description="A concise narrative summary of the project and its key findings.",
    )

    # Portfolio metadata tags (BMRC routing guide) for peer filtering and benchmarking
    study_stage: Optional[str] = Field(
        None,
        description="Study stage: Exploration, MRE, PEA, PFS, FS, or operating.",
    )
    deposit_type: Optional[str] = Field(
        None,
        description="Deposit type (e.g. porphyry, VMS, skarn, IOCG, sediment-hosted).",
    )
    mining_method: Optional[str] = Field(
        None,
        description="Primary mining method: open pit, underground, or combined.",
    )
    processing_route: Optional[str] = Field(
        None,
        description="Processing route (e.g. flotation, leach, CIL, heap leach).",
    )
    ore_type: Optional[str] = Field(
        None,
        description="Dominant ore type: oxide, transition, sulphide, or fresh.",
    )
    cutoff_type: Optional[str] = Field(
        None,
        description="Cut-off type: grade cut-off, NSR cut-off, or pit shell.",
    )
    economic_year: Optional[str] = Field(
        None,
        description="Year of metal price and cost assumptions used in economics.",
    )
    effective_date: Optional[str] = Field(
        None,
        description="Effective date of the mineral resource or reserve statement.",
    )
    primary_commodity: Optional[str] = Field(
        None,
        description="Primary commodity symbol for peer matching (e.g. Cu, Au, Li).",
    )
