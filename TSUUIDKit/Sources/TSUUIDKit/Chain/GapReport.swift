import Foundation

/// A detected gap in a knowledge chain.
public struct GapReport: Sendable {
    public let position: Int
    public let beforeUUID: UUID
    public let afterUUID: UUID
    public let distance: Float
    public let expectedDistance: Float
    public let severity: Float
    public let inferredTrits: TritVector
}
