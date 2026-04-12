import SwiftUI
import TSUUIDKit
import SwiftyDropbox

@main
struct TSUUID768App: App {
    @StateObject private var knowledge = KnowledgeService()
    @StateObject private var models = ModelService()
    @StateObject private var sync = SyncService()

    init() {
        DropboxClientsManager.setupWithAppKey(SyncService.dropboxAppKey)
    }

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
                .onOpenURL { url in
                    // Handle Dropbox OAuth redirect
                    sync.handleAuthRedirect(url)
                }
                .onReceive(NotificationCenter.default.publisher(
                    for: UIApplication.didReceiveMemoryWarningNotification
                )) { _ in
                    models.handleMemoryWarning()
                }
        }
    }
}
