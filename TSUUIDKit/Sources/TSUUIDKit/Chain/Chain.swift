import Foundation

/// Self-assembling semantic knowledge chain.
/// Port of Python tsuuid.chain.Chain.
public struct Chain: Sendable {
    public let chainId: String
    public let domain: String
    public private(set) var links: [ChainLink] = []

    public init(chainId: String = "", domain: String = "") {
        self.chainId = chainId.isEmpty ? String(UUID().uuidString.prefix(8)) : chainId
        self.domain = domain
    }

    public var count: Int { links.count }
    public subscript(index: Int) -> ChainLink { links[index] }

    public mutating func append(_ link: ChainLink) {
        var newLink = link
        newLink.chainId = chainId
        newLink.domain = domain.isEmpty ? link.domain : domain
        newLink.part = links.count + 1

        if !links.isEmpty {
            links[links.count - 1].nextUUID = newLink.uuid
            newLink.prevUUID = links[links.count - 1].uuid
        }
        links.append(newLink)
    }

    public mutating func insert(at position: Int, link: ChainLink) {
        var newLink = link
        newLink.chainId = chainId
        newLink.domain = domain.isEmpty ? link.domain : domain
        links.insert(newLink, at: position)
        rebuildPointers()
    }

    private mutating func rebuildPointers() {
        for i in 0..<links.count {
            links[i].part = i + 1
            links[i].total = links.count
            links[i].prevUUID = i > 0 ? links[i - 1].uuid : nil
            links[i].nextUUID = i < links.count - 1 ? links[i + 1].uuid : nil
        }
    }

    public func distances(metric: Compose.DistanceMetric = .cosine) -> [Float] {
        guard links.count >= 2 else { return [] }
        return (0..<links.count - 1).map {
            links[$0].distance(to: links[$0 + 1], metric: metric)
        }
    }

    public func validateCoherence(metric: Compose.DistanceMetric = .cosine)
        -> [(Int, Float, String)] {
        let dists = distances(metric: metric)
        guard !dists.isEmpty else { return [] }

        let sorted = dists.sorted()
        let median = sorted[sorted.count / 2]
        let mean = dists.reduce(0, +) / Float(dists.count)
        let variance = dists.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(dists.count)
        let std = dists.count > 2 ? sqrt(variance) : median * 0.3

        return dists.enumerated().map { i, d in
            let status: String
            if std < 1e-6 { status = "ok" }
            else if d > median + 2 * std { status = "break" }
            else if d > median + std { status = "weak" }
            else { status = "ok" }
            return (i + 1, d, status)
        }
    }

    public func detectGaps(metric: Compose.DistanceMetric = .cosine) -> [GapReport] {
        let dists = distances(metric: metric)
        guard dists.count >= 3 else { return [] }

        let sorted = dists.sorted()
        let median = sorted[sorted.count / 2]
        let mean = dists.reduce(0, +) / Float(dists.count)
        let variance = dists.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(dists.count)
        let std = sqrt(variance)
        let threshold = median + 1.5 * std

        var gaps: [GapReport] = []
        for (i, d) in dists.enumerated() where d > threshold && std > 1e-6 {
            let before = links[i].trits.trits
            let after = links[i + 1].trits.trits
            let midpoint = zip(before, after).map { a, b -> Int8 in
                let sum = Int16(a) + Int16(b)
                if sum > 0 { return 1 }
                if sum < 0 { return -1 }
                return 0
            }
            gaps.append(GapReport(
                position: i + 1,
                beforeUUID: links[i].uuid,
                afterUUID: links[i + 1].uuid,
                distance: d,
                expectedDistance: median,
                severity: (d - median) / std,
                inferredTrits: TritVector(trits: midpoint)
            ))
        }
        return gaps
    }

    public func toUUIDs() -> [UUID] { links.map(\.uuid) }

    public func toDict() -> [String: Any] {
        [
            "chain_id": chainId,
            "domain": domain,
            "length": links.count,
            "links": links.map { link in
                [
                    "uuid": link.uuid.uuidString,
                    "trits": link.trits.display(),
                    "prev": link.prevUUID?.uuidString as Any,
                    "next": link.nextUUID?.uuidString as Any,
                    "part": link.part,
                    "source": link.source,
                    "domain": link.domain,
                    "created_at": link.createdAt,
                ] as [String: Any]
            },
        ]
    }

    public func summary() -> String {
        let gaps = detectGaps()
        let coherence = validateCoherence()
        let breaks = coherence.filter { $0.2 == "break" }.count
        return "Chain[\(chainId)] domain=\(domain) links=\(links.count) gaps=\(gaps.count) breaks=\(breaks)"
    }
}
