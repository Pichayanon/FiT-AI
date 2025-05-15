import Foundation
import AVFoundation
import Combine
import SwiftUI

class CameraManager: NSObject, ObservableObject {
    @Published var keypointSequence: [[Float]] = []
    let session = AVCaptureSession()

    override init() {
        super.init()
        configureSession()
        startMockSequence()
    }

    func startMockSequence() {
        keypointSequence = mockSequences.randomElement() ?? []
    }

    func resetSequence() {
        keypointSequence.removeAll()
        startMockSequence()
    }

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .medium

        guard let camera = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: camera),
              session.canAddInput(input) else {
            print("Failed to access camera input")
            return
        }

        session.addInput(input)
        session.commitConfiguration()
        session.startRunning()
    }
}
