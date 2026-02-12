import Foundation
import AVFoundation
import Combine
import CoreImage
import UIKit

final class CameraManager: NSObject, ObservableObject {
    let session = AVCaptureSession()

    @Published var isRunning: Bool = false
    @Published var lastError: String? = nil

    var onBackendMessage: ((String) -> Void)?

    private let videoOutput = AVCaptureVideoDataOutput()
    private let videoQueue = DispatchQueue(label: "camera.video.queue")
    private let ciContext = CIContext()

    private var ws: VideoWebSocketClient?
    private var lastSentTime: CFTimeInterval = 0
    private let targetFPS: Double = 10

    override init() {
        super.init()
        configureSession()
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .high

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

    func startSession() {
        guard !session.isRunning else { return }
        videoQueue.async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async {
                self?.isRunning = true
            }
        }
    }

    func stopSession() {
        guard session.isRunning else { return }
        videoQueue.async { [weak self] in
            self?.session.stopRunning()
            DispatchQueue.main.async {
                self?.isRunning = false
            }
        }
    }

    func startStreaming(to urlString: String) {
        guard let url = URL(string: urlString) else {
            lastError = "Invalid WS URL"
            return
        }

        let client = VideoWebSocketClient(config: .init(url: url))
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

    func stopStreaming() {
        sendJSON(["type": "stop", "ts": Int(Date().timeIntervalSince1970 * 1000)])
        ws?.disconnect()
        ws = nil
    }

    private func sendJSON(_ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []),
              let text = String(data: data, encoding: .utf8) else { return }
        ws?.sendText(text)
    }

    private func shouldSendFrame(now: CFTimeInterval) -> Bool {
        let interval = 1.0 / targetFPS
        if now - lastSentTime >= interval {
            lastSentTime = now
            return true
        }
        return false
    }

    private func jpegBase64(from pixelBuffer: CVPixelBuffer) -> String? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)

        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        let uiImage = UIImage(cgImage: cgImage)

        guard let jpegData = uiImage.jpegData(compressionQuality: 0.45) else { return nil }
        return jpegData.base64EncodedString()
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {

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
