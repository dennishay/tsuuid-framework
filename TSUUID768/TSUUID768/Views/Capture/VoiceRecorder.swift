import SwiftUI
import Speech
import AVFoundation

/// On-device voice recording + speech-to-text via Apple Speech framework.
/// Holds the mic recording while `isRecording` is true, streams transcription as you speak.
@MainActor
final class VoiceRecorder: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var transcript = ""
    @Published var errorMessage: String?

    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    /// Request permissions. Call before start().
    func requestPermissions() async -> Bool {
        // Speech recognition permission
        let speechAuth = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }

        guard speechAuth == .authorized else {
            errorMessage = "Speech recognition permission denied"
            return false
        }

        // Microphone permission
        let micAuth = await AVAudioApplication.requestRecordPermission()
        guard micAuth else {
            errorMessage = "Microphone permission denied"
            return false
        }

        return true
    }

    func start() throws {
        guard !isRecording else { return }

        transcript = ""
        errorMessage = nil

        // Cancel any ongoing task
        recognitionTask?.cancel()
        recognitionTask = nil

        // Audio session
        let audioSession = AVAudioSession.sharedInstance()
        try audioSession.setCategory(.record, mode: .measurement, options: .duckOthers)
        try audioSession.setActive(true, options: .notifyOthersOnDeactivation)

        // Recognition request (prefer on-device)
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if #available(iOS 13.0, *) {
            request.requiresOnDeviceRecognition = true
        }
        recognitionRequest = request

        // Audio engine
        let engine = AVAudioEngine()
        audioEngine = engine
        let inputNode = engine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { buffer, _ in
            request.append(buffer)
        }

        engine.prepare()
        try engine.start()

        // Start recognition
        recognitionTask = recognizer?.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                if let result = result {
                    self?.transcript = result.bestTranscription.formattedString
                }
                if error != nil || result?.isFinal == true {
                    self?.stop()
                }
            }
        }

        isRecording = true
    }

    func stop() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()

        audioEngine = nil
        recognitionRequest = nil
        recognitionTask = nil

        isRecording = false

        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
}
