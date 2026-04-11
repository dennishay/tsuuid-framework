import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            SearchTab()
                .tabItem {
                    Label("Search", systemImage: "magnifyingglass")
                }

            CaptureTab()
                .tabItem {
                    Label("Capture", systemImage: "camera")
                }

            ChainsTab()
                .tabItem {
                    Label("Chains", systemImage: "link")
                }

            SyncTab()
                .tabItem {
                    Label("Sync", systemImage: "arrow.triangle.2.circlepath")
                }
        }
    }
}
