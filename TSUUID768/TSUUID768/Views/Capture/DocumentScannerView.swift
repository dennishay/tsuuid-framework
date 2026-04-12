import SwiftUI
import VisionKit
import Vision

/// Wraps VNDocumentCameraViewController for SwiftUI.
struct DocumentScannerView: UIViewControllerRepresentable {
    let onFinish: (String) -> Void  // returns OCR'd text
    let onCancel: () -> Void

    func makeUIViewController(context: Context) -> VNDocumentCameraViewController {
        let vc = VNDocumentCameraViewController()
        vc.delegate = context.coordinator
        return vc
    }

    func updateUIViewController(_ uiViewController: VNDocumentCameraViewController, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator(onFinish: onFinish, onCancel: onCancel)
    }

    class Coordinator: NSObject, VNDocumentCameraViewControllerDelegate {
        let onFinish: (String) -> Void
        let onCancel: () -> Void

        init(onFinish: @escaping (String) -> Void, onCancel: @escaping () -> Void) {
            self.onFinish = onFinish
            self.onCancel = onCancel
        }

        func documentCameraViewController(_ controller: VNDocumentCameraViewController,
                                          didFinishWith scan: VNDocumentCameraScan) {
            // OCR all pages using Vision
            var allText: [String] = []
            let group = DispatchGroup()

            for pageIndex in 0..<scan.pageCount {
                let image = scan.imageOfPage(at: pageIndex)
                guard let cgImage = image.cgImage else { continue }

                group.enter()
                let request = VNRecognizeTextRequest { req, _ in
                    defer { group.leave() }
                    guard let observations = req.results as? [VNRecognizedTextObservation] else { return }
                    let pageText = observations
                        .compactMap { $0.topCandidates(1).first?.string }
                        .joined(separator: "\n")
                    allText.append(pageText)
                }
                request.recognitionLevel = .accurate
                request.usesLanguageCorrection = true

                let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
                try? handler.perform([request])
            }

            group.notify(queue: .main) {
                let combined = allText.joined(separator: "\n\n")
                self.onFinish(combined)
                controller.dismiss(animated: true)
            }
        }

        func documentCameraViewControllerDidCancel(_ controller: VNDocumentCameraViewController) {
            onCancel()
            controller.dismiss(animated: true)
        }

        func documentCameraViewController(_ controller: VNDocumentCameraViewController,
                                          didFailWithError error: Error) {
            onCancel()
            controller.dismiss(animated: true)
        }
    }
}
