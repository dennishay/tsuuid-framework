import SwiftUI
import TSUUIDKit

struct SearchTab: View {
    @EnvironmentObject var knowledge: KnowledgeService
    @EnvironmentObject var model: ModelService
    @StateObject private var vm = SearchViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                HStack {
                    Image(systemName: "magnifyingglass")
                        .foregroundStyle(.secondary)
                    TextField("Search knowledge...", text: $vm.query)
                        .textFieldStyle(.plain)
                        .autocorrectionDisabled()
                        .onChange(of: vm.query) { _, _ in
                            vm.search(using: knowledge, model: model)
                        }
                    if !vm.query.isEmpty {
                        Button { vm.query = "" } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding()
                .background(.ultraThinMaterial)

                if vm.isSearching {
                    ProgressView()
                        .padding()
                }

                if vm.results.isEmpty && !vm.query.isEmpty && !vm.isSearching {
                    ContentUnavailableView("No results",
                        systemImage: "magnifyingglass",
                        description: Text("Try a different query"))
                } else {
                    List(vm.results, id: \.index) { result in
                        NavigationLink {
                            SearchDetailView(result: result)
                        } label: {
                            SearchResultRow(result: result)
                        }
                    }
                    .listStyle(.plain)
                }
            }
            .onTapGesture {
                UIApplication.shared.sendAction(
                    #selector(UIResponder.resignFirstResponder),
                    to: nil, from: nil, for: nil
                )
            }
            .navigationTitle("Search")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Text("\(knowledge.vectorCount)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }
}
