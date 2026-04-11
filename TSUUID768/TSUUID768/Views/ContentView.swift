import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            SearchTab()
                .tabItem {
                    Label("Search", systemImage: "magnifyingglass")
                }
                .tag(0)

            CaptureTab()
                .tabItem {
                    Label("Capture", systemImage: "camera")
                }
                .tag(1)

            ChainsTab()
                .tabItem {
                    Label("Chains", systemImage: "link")
                }
                .tag(2)

            SyncTab()
                .tabItem {
                    Label("Sync", systemImage: "arrow.triangle.2.circlepath")
                }
                .tag(3)
        }
        .onChange(of: selectedTab) { _, _ in
            // Dismiss keyboard when switching tabs
            UIApplication.shared.sendAction(
                #selector(UIResponder.resignFirstResponder),
                to: nil, from: nil, for: nil
            )
        }
    }
}
