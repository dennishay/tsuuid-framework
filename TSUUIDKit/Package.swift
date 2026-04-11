// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "TSUUIDKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14),
    ],
    products: [
        .library(name: "TSUUIDKit", targets: ["TSUUIDKit"]),
    ],
    targets: [
        .target(
            name: "TSUUIDKit",
            path: "Sources/TSUUIDKit"
        ),
        .testTarget(
            name: "TSUUIDKitTests",
            dependencies: ["TSUUIDKit"],
            path: "Tests/TSUUIDKitTests"
        ),
    ]
)
