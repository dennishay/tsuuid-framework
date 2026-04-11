import Foundation

/// A single semantic dimension axis.
public struct Axis: Sendable {
    public let id: Int
    public let name: String
    public let negative: String
    public let neutral: String
    public let positive: String
    public let layer: Layer

    public enum Layer: String, Sendable, CaseIterable {
        case protocol_ = "protocol"
        case hardware = "hardware"
        case organization = "organization"
        case application = "application"
        case entity = "entity"
        case field = "field"
    }

    public func describeTrit(_ value: Int8) -> String {
        switch value {
        case -1: return "\(name): \(negative)"
        case  0: return "\(name): \(neutral)"
        case  1: return "\(name): \(positive)"
        default: fatalError("Invalid trit value: \(value)")
        }
    }
}

/// The 81-dimensional meaning space.
/// Port of Python tsuuid.dimensions — identical axis definitions.
public enum SemanticDimensions {
    private static let protocolAxes: [Axis] = [
        Axis(id: 1,  name: "temporality",    negative: "past",       neutral: "atemporal",      positive: "future",        layer: .protocol_),
        Axis(id: 2,  name: "modality",        negative: "deletion",    neutral: "reference",      positive: "creation",      layer: .protocol_),
        Axis(id: 3,  name: "certainty",       negative: "denied",      neutral: "unknown",        positive: "confirmed",     layer: .protocol_),
        Axis(id: 4,  name: "cardinality",     negative: "singular",    neutral: "unspecified",    positive: "plural",        layer: .protocol_),
        Axis(id: 5,  name: "obligation",      negative: "forbidden",   neutral: "optional",       positive: "required",      layer: .protocol_),
        Axis(id: 6,  name: "visibility",      negative: "private",     neutral: "contextual",     positive: "public",        layer: .protocol_),
        Axis(id: 7,  name: "mutability",      negative: "immutable",   neutral: "unspecified",    positive: "mutable",       layer: .protocol_),
        Axis(id: 8,  name: "directionality",  negative: "inbound",     neutral: "undirected",     positive: "outbound",      layer: .protocol_),
        Axis(id: 9,  name: "granularity",     negative: "aggregate",   neutral: "unspecified",    positive: "atomic",        layer: .protocol_),
        Axis(id: 10, name: "lifecycle",        negative: "archived",    neutral: "active",         positive: "draft",         layer: .protocol_),
    ]

    private static let hardwareAxes: [Axis] = [
        Axis(id: 11, name: "compute_class",   negative: "embedded",    neutral: "general",        positive: "server",        layer: .hardware),
        Axis(id: 12, name: "connectivity",     negative: "offline",     neutral: "intermittent",   positive: "always-on",     layer: .hardware),
        Axis(id: 13, name: "storage_class",    negative: "ephemeral",   neutral: "cached",         positive: "persistent",    layer: .hardware),
        Axis(id: 14, name: "precision",        negative: "approximate", neutral: "standard",       positive: "exact",         layer: .hardware),
        Axis(id: 15, name: "endianness",       negative: "little",      neutral: "unspecified",    positive: "big",           layer: .hardware),
        Axis(id: 16, name: "word_width",       negative: "narrow",      neutral: "standard",       positive: "wide",          layer: .hardware),
        Axis(id: 17, name: "security_level",   negative: "untrusted",   neutral: "standard",       positive: "hardened",      layer: .hardware),
        Axis(id: 18, name: "power_state",      negative: "sleep",       neutral: "idle",           positive: "active",        layer: .hardware),
        Axis(id: 19, name: "locality",         negative: "remote",      neutral: "networked",      positive: "local",         layer: .hardware),
        Axis(id: 20, name: "determinism",      negative: "stochastic",  neutral: "unspecified",    positive: "deterministic", layer: .hardware),
    ]

    private static let organizationAxes: [Axis] = [
        Axis(id: 21, name: "ownership",        negative: "external",      neutral: "shared",           positive: "internal",      layer: .organization),
        Axis(id: 22, name: "confidentiality",  negative: "classified",    neutral: "standard",         positive: "open",          layer: .organization),
        Axis(id: 23, name: "authority",         negative: "subordinate",   neutral: "peer",             positive: "authoritative", layer: .organization),
        Axis(id: 24, name: "compliance",        negative: "non-compliant", neutral: "unassessed",       positive: "compliant",     layer: .organization),
        Axis(id: 25, name: "cost_center",       negative: "expense",       neutral: "neutral",          positive: "revenue",       layer: .organization),
        Axis(id: 26, name: "urgency",           negative: "deferred",      neutral: "normal",           positive: "critical",      layer: .organization),
        Axis(id: 27, name: "approval_state",    negative: "rejected",      neutral: "pending",          positive: "approved",      layer: .organization),
        Axis(id: 28, name: "department_class",  negative: "support",       neutral: "cross-functional", positive: "operations",    layer: .organization),
        Axis(id: 29, name: "geographic_scope",  negative: "local",         neutral: "regional",         positive: "global",        layer: .organization),
        Axis(id: 30, name: "retention",         negative: "delete",        neutral: "review",           positive: "retain",        layer: .organization),
        Axis(id: 31, name: "audit_status",      negative: "flagged",       neutral: "unaudited",        positive: "cleared",       layer: .organization),
        Axis(id: 32, name: "stakeholder",       negative: "external",      neutral: "mixed",            positive: "internal",      layer: .organization),
        Axis(id: 33, name: "maturity",          negative: "experimental",  neutral: "standard",         positive: "proven",        layer: .organization),
        Axis(id: 34, name: "impact",            negative: "low",           neutral: "moderate",         positive: "high",          layer: .organization),
        Axis(id: 35, name: "frequency",         negative: "rare",          neutral: "periodic",         positive: "continuous",    layer: .organization),
    ]

    private static let applicationAxes: [Axis] = [
        Axis(id: 36, name: "domain",           negative: "technical",     neutral: "general",          positive: "business",      layer: .application),
        Axis(id: 37, name: "data_type",        negative: "unstructured",  neutral: "semi-structured",  positive: "structured",    layer: .application),
        Axis(id: 38, name: "flow_direction",   negative: "outflow",       neutral: "static",           positive: "inflow",        layer: .application),
        Axis(id: 39, name: "value_sign",       negative: "negative",      neutral: "zero",             positive: "positive",      layer: .application),
        Axis(id: 40, name: "scale",            negative: "small",         neutral: "medium",           positive: "large",         layer: .application),
        Axis(id: 41, name: "relationship",     negative: "child",         neutral: "standalone",       positive: "parent",        layer: .application),
        Axis(id: 42, name: "status",           negative: "failed",        neutral: "in-progress",      positive: "complete",      layer: .application),
        Axis(id: 43, name: "priority",         negative: "low",           neutral: "normal",           positive: "high",          layer: .application),
        Axis(id: 44, name: "source_type",      negative: "computed",      neutral: "hybrid",           positive: "observed",      layer: .application),
        Axis(id: 45, name: "currency",         negative: "stale",         neutral: "current",          positive: "projected",     layer: .application),
        Axis(id: 46, name: "aggregation",      negative: "detail",        neutral: "summary",          positive: "rollup",        layer: .application),
        Axis(id: 47, name: "sentiment",        negative: "negative",      neutral: "neutral",          positive: "positive",      layer: .application),
        Axis(id: 48, name: "action_type",      negative: "query",         neutral: "reference",        positive: "mutation",      layer: .application),
        Axis(id: 49, name: "entity_class",     negative: "event",         neutral: "abstract",         positive: "physical",      layer: .application),
        Axis(id: 50, name: "quantity_type",    negative: "rate",          neutral: "absolute",         positive: "cumulative",    layer: .application),
        Axis(id: 51, name: "time_horizon",     negative: "historical",    neutral: "current",          positive: "forecast",      layer: .application),
        Axis(id: 52, name: "validation",       negative: "invalid",       neutral: "unvalidated",      positive: "validated",     layer: .application),
        Axis(id: 53, name: "version_state",    negative: "deprecated",    neutral: "current",          positive: "next",          layer: .application),
        Axis(id: 54, name: "dependency",       negative: "blocked",       neutral: "independent",      positive: "blocking",      layer: .application),
        Axis(id: 55, name: "format_class",     negative: "binary",        neutral: "mixed",            positive: "text",          layer: .application),
    ]

    private static let entityAxes: [Axis] = [
        Axis(id: 56, name: "entity_role",      negative: "object",        neutral: "attribute",        positive: "subject",       layer: .entity),
        Axis(id: 57, name: "identity_type",    negative: "anonymous",     neutral: "pseudonymous",     positive: "identified",    layer: .entity),
        Axis(id: 58, name: "completeness",     negative: "partial",       neutral: "standard",         positive: "complete",      layer: .entity),
        Axis(id: 59, name: "encoding",         negative: "compressed",    neutral: "standard",         positive: "expanded",      layer: .entity),
        Axis(id: 60, name: "reference_type",   negative: "symbolic",      neutral: "direct",           positive: "semantic",      layer: .entity),
        Axis(id: 61, name: "nullability",      negative: "null",          neutral: "default",          positive: "populated",     layer: .entity),
        Axis(id: 62, name: "uniqueness",       negative: "duplicate",     neutral: "unverified",       positive: "unique",        layer: .entity),
        Axis(id: 63, name: "ordering",         negative: "unordered",     neutral: "partial",          positive: "total",         layer: .entity),
        Axis(id: 64, name: "indexability",     negative: "unindexed",     neutral: "secondary",        positive: "primary",       layer: .entity),
        Axis(id: 65, name: "volatility",       negative: "stable",        neutral: "moderate",         positive: "volatile",      layer: .entity),
        Axis(id: 66, name: "precision_level",  negative: "approximate",   neutral: "standard",         positive: "high-precision",layer: .entity),
        Axis(id: 67, name: "origin",           negative: "derived",       neutral: "hybrid",           positive: "original",      layer: .entity),
        Axis(id: 68, name: "size_class",       negative: "small",         neutral: "medium",           positive: "large",         layer: .entity),
        Axis(id: 69, name: "link_density",     negative: "isolated",      neutral: "sparse",           positive: "dense",         layer: .entity),
        Axis(id: 70, name: "update_mode",      negative: "batch",         neutral: "periodic",         positive: "realtime",      layer: .entity),
    ]

    private static let fieldAxes: [Axis] = [
        Axis(id: 71, name: "numeric_sign",     negative: "negative",      neutral: "zero",             positive: "positive",      layer: .field),
        Axis(id: 72, name: "bound_type",       negative: "lower",         neutral: "unbounded",        positive: "upper",         layer: .field),
        Axis(id: 73, name: "comparison",       negative: "less_than",     neutral: "equal",            positive: "greater_than",  layer: .field),
        Axis(id: 74, name: "trend",            negative: "decreasing",    neutral: "stable",           positive: "increasing",    layer: .field),
        Axis(id: 75, name: "anomaly",          negative: "below_normal",  neutral: "normal",           positive: "above_normal",  layer: .field),
        Axis(id: 76, name: "confidence",       negative: "low",           neutral: "medium",           positive: "high",          layer: .field),
        Axis(id: 77, name: "relevance",        negative: "tangential",    neutral: "moderate",         positive: "primary",       layer: .field),
        Axis(id: 78, name: "recency",          negative: "old",           neutral: "moderate",         positive: "new",           layer: .field),
        Axis(id: 79, name: "specificity",      negative: "general",       neutral: "moderate",         positive: "specific",      layer: .field),
        Axis(id: 80, name: "affect",           negative: "negative",      neutral: "neutral",          positive: "positive",      layer: .field),
        Axis(id: 81, name: "salience",         negative: "background",    neutral: "normal",           positive: "foreground",    layer: .field),
    ]

    public static let allAxes: [Axis] =
        protocolAxes + hardwareAxes + organizationAxes +
        applicationAxes + entityAxes + fieldAxes

    public static func axis(_ id: Int) -> Axis { allAxes[id - 1] }

    public static func axes(for layer: Axis.Layer) -> [Axis] {
        allAxes.filter { $0.layer == layer }
    }

    public static func describe(_ tv: TritVector) -> [String] {
        tv.trits.enumerated().compactMap { i, t in
            guard t != 0, i < allAxes.count else { return nil }
            return allAxes[i].describeTrit(t)
        }
    }

    public static func layerSummary(_ tv: TritVector) -> [Axis.Layer: [String]] {
        var result: [Axis.Layer: [String]] = [:]
        for (i, t) in tv.trits.enumerated() where t != 0 && i < allAxes.count {
            let ax = allAxes[i]
            result[ax.layer, default: []].append(ax.describeTrit(t))
        }
        return result
    }
}
