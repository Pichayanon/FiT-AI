import Foundation
import AVFoundation
import Combine
import CoreImage
import UIKit

/// Writes camera sample buffers to a local `.mov` file for later playback.
private final class SessionVideoRecorder {
    private enum State {
        case idle
        case preparing(URL)
        case recording(URL)
        case finishing
    }

    private var state: State = .idle
    private var writer: AVAssetWriter?
    private var writerInput: AVAssetWriterInput?
    private var hasWrittenFrame = false

    /// Starts a new recording that will be initialized on the next sample buffer.
    func start(to url: URL) {
        reset(deleteCurrentFile: true)
        deleteFile(at: url)
        state = .preparing(url)
    }

    /// Appends a captured frame to the active recording, if there is one.
    func append(_ sampleBuffer: CMSampleBuffer) {
        switch state {
        case .idle, .finishing:
            return
        case .preparing(let url):
            guard prepareWriter(for: sampleBuffer, url: url) else {
                reset(deleteCurrentFile: true)
                return
            }
            state = .recording(url)
            fallthrough
        case .recording:
            appendToWriter(sampleBuffer)
        }
    }

    /// Finalizes the active recording.
    func finish(keepFile: Bool, completion: @escaping (URL?) -> Void) {
        let completionOnMain: (URL?) -> Void = { url in
            DispatchQueue.main.async {
                completion(url)
            }
        }

        let outputURL = currentURL

        guard let outputURL else {
            reset(deleteCurrentFile: true)
            completionOnMain(nil)
            return
        }

        guard let writer,
              let writerInput,
              hasWrittenFrame else {
            reset(deleteCurrentFile: true)
            if !keepFile {
                deleteFile(at: outputURL)
            }
            completionOnMain(nil)
            return
        }

        state = .finishing

        if !keepFile {
            writer.cancelWriting()
            reset(deleteCurrentFile: true)
            completionOnMain(nil)
            return
        }

        writerInput.markAsFinished()
        writer.finishWriting { [weak self] in
            let finishedURL: URL?
            if writer.status == .completed {
                finishedURL = outputURL
            } else {
                self?.deleteFile(at: outputURL)
                finishedURL = nil
            }
            self?.reset(deleteCurrentFile: false)
            completionOnMain(finishedURL)
        }
    }

    private var currentURL: URL? {
        switch state {
        case .idle, .finishing:
            return writer?.outputURL
        case .preparing(let url), .recording(let url):
            return url
        }
    }

    /// Builds the writer lazily once we know the video dimensions.
    private func prepareWriter(for sampleBuffer: CMSampleBuffer, url: URL) -> Bool {
        guard let formatDescription = CMSampleBufferGetFormatDescription(sampleBuffer) else { return false }

        let dimensions = CMVideoFormatDescriptionGetDimensions(formatDescription)
        let width = max(Int(dimensions.width), 1)
        let height = max(Int(dimensions.height), 1)

        do {
            let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
            let outputSettings: [String: Any] = [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: max(width * height * 8, 2_000_000),
                    AVVideoExpectedSourceFrameRateKey: 30
                ]
            ]
            let writerInput = AVAssetWriterInput(
                mediaType: .video,
                outputSettings: outputSettings,
                sourceFormatHint: formatDescription
            )
            writerInput.expectsMediaDataInRealTime = true

            guard writer.canAdd(writerInput) else {
                return false
            }

            writer.add(writerInput)
            self.writer = writer
            self.writerInput = writerInput
            self.hasWrittenFrame = false
            return true
        } catch {
            print("[SessionVideoRecorder] Prepare writer error: \(error)")
            return false
        }
    }

    /// Writes a single sample buffer to the current asset writer.
    private func appendToWriter(_ sampleBuffer: CMSampleBuffer) {
        guard let writer,
              let writerInput else { return }

        if writer.status == .unknown {
            let startTime = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
            guard writer.startWriting() else {
                print("[SessionVideoRecorder] startWriting failed: \(writer.error?.localizedDescription ?? "unknown")")
                reset(deleteCurrentFile: true)
                return
            }
            writer.startSession(atSourceTime: startTime)
        }

        guard writer.status == .writing else {
            if writer.status == .failed {
                print("[SessionVideoRecorder] Writer failed: \(writer.error?.localizedDescription ?? "unknown")")
            }
            reset(deleteCurrentFile: true)
            return
        }

        guard writerInput.isReadyForMoreMediaData else { return }

        if writerInput.append(sampleBuffer) {
            hasWrittenFrame = true
        } else {
            print("[SessionVideoRecorder] Append failed: \(writer.error?.localizedDescription ?? "unknown")")
            reset(deleteCurrentFile: true)
        }
    }

    /// Resets all recorder state and optionally deletes the current output file.
    private func reset(deleteCurrentFile: Bool) {
        let urlToDelete = deleteCurrentFile ? currentURL : nil
        writerInput = nil
        writer = nil
        hasWrittenFrame = false
        state = .idle

        if let urlToDelete {
            deleteFile(at: urlToDelete)
        }
    }

    private func deleteFile(at url: URL) {
        guard FileManager.default.fileExists(atPath: url.path) else { return }

        do {
            try FileManager.default.removeItem(at: url)
        } catch {
            print("[SessionVideoRecorder] Delete file error: \(error)")
        }
    }
}

/// Manages the device camera session and streams JPEG frames to a backend via WebSocket.
///
/// Lifecycle:
/// 1. `startSession()` — begins the AVCaptureSession (camera preview).
/// 2. `startStreaming(to:)` — opens a WebSocket and starts sending frames at `targetFPS`.
/// 3. `stopStreaming()` — closes the WebSocket; camera keeps running.
/// 4. `stopSession()` — stops both camera and streaming.
///
/// Backend responses are forwarded through the `onBackendMessage` callback.
final class CameraService: NSObject, ObservableObject {
    let session = AVCaptureSession()

    @Published var isRunning: Bool = false
    @Published var lastError: String?

    /// Called on the main queue whenever the backend sends a text message.
    var onBackendMessage: ((String) -> Void)?

    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "camera.video.queue")
    private let ciContext = CIContext()
    private let recorder = SessionVideoRecorder()

    private var ws: WebSocketService?
    private var lastSentTime: CFTimeInterval = 0
    private let targetFPS: Double = 30

    override init() {
        super.init()
        configureSession()
    }

    // MARK: - Session Configuration

    /// Sets up the capture session with front camera input and video output.
    private func configureSession() {
        session.beginConfiguration()
        if session.canSetSessionPreset(.medium) {
            session.sessionPreset = .medium
        } else {
            session.sessionPreset = .high
        }

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) ??
                          AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            lastError = "No camera device available"
            session.commitConfiguration()
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) { session.addInput(input) }
        } catch {
            lastError = "Camera input error: \(error.localizedDescription)"
            session.commitConfiguration()
            return
        }

        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)

        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        if let conn = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if conn.isVideoRotationAngleSupported(90) {
                    conn.videoRotationAngle = 90
                }
            } else {
                if conn.isVideoOrientationSupported {
                    conn.videoOrientation = .portrait
                }
            }
            if conn.isVideoMirroringSupported {
                conn.isVideoMirrored = true
            }
        }

        session.commitConfiguration()
    }

    // MARK: - Session Lifecycle

    /// Starts the camera capture session on a background queue.
    func startSession() {
        guard !session.isRunning else { return }
        videoQueue.async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async {
                self?.isRunning = true
            }
        }
    }

    /// Stops the camera capture session.
    func stopSession() {
        videoQueue.async { [weak self] in
            guard let self else { return }

            if self.session.isRunning {
                self.session.stopRunning()
            }

            DispatchQueue.main.async {
                self.isRunning = false
            }
        }
    }

    /// Stops the camera capture session and optionally discards any unfinished recording.
    func stopSession(discardRecording: Bool) {
        videoQueue.async { [weak self] in
            guard let self else { return }

            if discardRecording {
                self.recorder.finish(keepFile: false) { _ in }
            }

            if self.session.isRunning {
                self.session.stopRunning()
            }

            DispatchQueue.main.async {
                self.isRunning = false
            }
        }
    }

    // MARK: - WebSocket Streaming

    /// Opens a WebSocket connection and begins streaming frames to the given URL.
    func startStreaming(to urlString: String) {
        guard let url = URL(string: urlString) else {
            lastError = "Invalid WS URL"
            return
        }

        let client = WebSocketService(config: .init(url: url))
        client.onTextMessage = { [weak self] text in
            self?.onBackendMessage?(text)
        }
        client.onDisconnected = { [weak self] err in
            DispatchQueue.main.async {
                self?.lastError = err?.localizedDescription
            }
        }
        self.ws = client
        client.connect()

        sendJSON(["type": "start", "ts": Int(Date().timeIntervalSince1970 * 1000)])
    }

    /// Sends a stop message and disconnects the WebSocket.
    func stopStreaming() {
        sendJSON(["type": "stop", "ts": Int(Date().timeIntervalSince1970 * 1000)])
        ws?.disconnect()
        ws = nil
    }

    /// Begins recording the current camera feed to a local video file.
    func beginSessionRecording() {
        videoQueue.async { [weak self] in
            guard let self else { return }
            guard let outputURL = SessionVideoStore.makeRecordingURL() else {
                DispatchQueue.main.async {
                    self.lastError = "Could not create session recording file"
                }
                return
            }

            self.recorder.start(to: outputURL)
        }
    }

    /// Finalizes the current local session recording.
    func finishSessionRecording(keepFile: Bool = true, completion: @escaping (URL?) -> Void) {
        videoQueue.async { [weak self] in
            guard let self else {
                DispatchQueue.main.async {
                    completion(nil)
                }
                return
            }

            self.recorder.finish(keepFile: keepFile, completion: completion)
        }
    }

    // MARK: - Private Helpers

    /// Serializes a dictionary to JSON and sends it as a text WebSocket message.
    private func sendJSON(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
              let text = String(data: data, encoding: .utf8) else { return }
        ws?.sendText(text)
    }

    /// Rate-limits frame sending to `targetFPS`.
    private func shouldSendFrame(now: CFTimeInterval) -> Bool {
        let interval = 1.0 / targetFPS
        if now - lastSentTime >= interval {
            lastSentTime = now
            return true
        }
        return false
    }

    /// Converts a pixel buffer to a base64-encoded JPEG string.
    private func jpegBase64(from pixelBuffer: CVPixelBuffer) -> String? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        let uiImage = UIImage(cgImage: cgImage)

        guard let jpegData = uiImage.jpegData(compressionQuality: 0.45) else { return nil }
        return jpegData.base64EncodedString()
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraService: AVCaptureVideoDataOutputSampleBufferDelegate {
    /// Called for each captured video frame. Rate-limits and sends frames to the backend.
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        recorder.append(sampleBuffer)

        guard ws != nil else { return }
        let now = CACurrentMediaTime()
        guard shouldSendFrame(now: now) else { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        guard let b64 = jpegBase64(from: pixelBuffer) else { return }

        let payload: [String: Any] = [
            "type": "frame",
            "ts": Int(Date().timeIntervalSince1970 * 1000),
            "jpeg_b64": b64
        ]
        sendJSON(payload)
    }
}
