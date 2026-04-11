import Foundation

/// A single gene in a knowledge chain.
/// The TSUUID (81 trits packed into UUID v8) IS the meaning.
public struct ChainLink: Sendable {
    public let uuid: UUID
    public let trits: TritVector
    public var vec768: Vector768?
    public var prevUUID: UUID?
    public var nextUUID: UUID?
    public var chainId: String
    public var part: Int
    public var total: Int?
    public var source: String
    public var domain: String
    public var createdAt: String

    public init(uuid: UUID, trits: TritVector, vec768: Vector768? = nil,
                prevUUID: UUID? = nil, nextUUID: UUID? = nil,
                chainId: String = "", part: Int = 0, total: Int? = nil,
                source: String = "", domain: String = "",
                createdAt: String? = nil) {
        self.uuid = uuid
        self.trits = trits
        self.vec768 = vec768
        self.prevUUID = prevUUID
        self.nextUUID = nextUUID
        self.chainId = chainId
        self.part = part
        self.total = total
        self.source = source
        self.domain = domain

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withFullDate, .withTime, .withColonSeparatorInTime]
        self.createdAt = createdAt ?? formatter.string(from: Date())
    }

    public static func fromUUID(_ uuid: UUID, source: String = "",
                                domain: String = "") -> ChainLink {
        let trits = Packing.unpackFromUUID(uuid)
        return ChainLink(uuid: uuid, trits: trits, source: source, domain: domain)
    }

    public func display() -> String { trits.display() }

    public func distance(to other: ChainLink,
                         metric: Compose.DistanceMetric = .cosine) -> Float {
        Compose.semanticDistance(uuid, other.uuid, metric: metric)
    }

    public func diff(to other: ChainLink) -> TritVector {
        Compose.diff(uuid, other.uuid)
    }
}
