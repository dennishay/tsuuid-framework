import SwiftUI
import TSUUIDKit

@main
struct TSUUID768App: App {
    @StateObject private var knowledge = KnowledgeService()
    @StateObject private var models = ModelService()
    @StateObject private var sync = SyncService()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(knowledge)
                .environmentObject(models)
                .environmentObject(sync)
                .preferredColorScheme(.dark)
                .task {
                    await knowledge.load()
                    await models.loadModels()
                }
                .onReceive(NotificationCenter.default.publisher(
                    for: UIApplication.didReceiveMemoryWarningNotification
                )) { _ in
                    models.handleMemoryWarning()
                }
        }
    }
}
