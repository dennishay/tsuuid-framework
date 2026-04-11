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
    dependencies: [
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "0.1.12"),
    ],
    targets: [
        .target(
            name: "TSUUIDKit",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/TSUUIDKit"
        ),
        .testTarget(
            name: "TSUUIDKitTests",
            dependencies: ["TSUUIDKit"],
            path: "Tests/TSUUIDKitTests"
        ),
    ]
)
