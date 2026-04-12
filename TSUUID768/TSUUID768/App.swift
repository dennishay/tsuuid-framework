import SwiftUI
import TSUUIDKit
import SwiftyDropbox

@main
struct TSUUID768App: App {
    @StateObject private var knowledge = KnowledgeService()
    @StateObject private var models = ModelService()
    @StateObject private var sync = SyncService()

    init() {
        // Setup Dropbox SDK once at app launch
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
                    sync.connect()  // safe to call — checks authorizedClient
                }
                .onOpenURL { url in
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
