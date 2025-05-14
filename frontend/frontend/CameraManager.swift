//import Foundation
//import AVFoundation
//import CoreML
//
//
//class CameraManager: NSObject, ObservableObject {
//    let session = AVCaptureSession()
//    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
//
//    @Published var feedback = "Waiting for squat..."
//    @Published var reps = 0
//    @Published var elapsedTime = 0
//
//    private var timer: Timer?
//    private var keypointSequence: [[Float]] = []
//
//    private let model = try? SquatClassifier(configuration: MLModelConfiguration())
//
//    override init() {
//        super.init()
//        setupSession()
//    }
//
//    func setupSession() {
//        session.beginConfiguration()
//
//        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera,
//                                                   for: .video,
//                                                   position: .front),
//              let input = try? AVCaptureDeviceInput(device: device),
//              session.canAddInput(input) else {
//            return
//        }
//        session.addInput(input)
//        session.commitConfiguration()
//    }
//
//    func startSession() {
//        sessionQueue.async {
//            if !self.session.isRunning {
//                self.session.startRunning()
//                DispatchQueue.main.async {
//                    self.startTimer()
//                }
//            }
//        }
//    }
//
//
//    func stopSession() {
//        sessionQueue.async {
//            self.session.stopRunning()
//            DispatchQueue.main.async {
//                self.stopTimer()
//            }
//        }
//    }
//
//    private func startTimer() {
//        elapsedTime = 0
//        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
//            self.elapsedTime += 1
//            // simulate a squat every 5 seconds (for testing)
//            if self.elapsedTime % 5 == 0 {
//                self.simulateSquatInput()
//            }
//        }
//    }
//
//    private func stopTimer() {
//        timer?.invalidate()
//        timer = nil
//    }
//
//    private func simulateSquatInput() {
//        let simulatedSequence = (0..<30).map { _ in (0..<48).map { _ in Float.random(in: 0...1) } }
//        guard let input = createMLMultiArray(from: simulatedSequence),
//              let model = model,
//              let output = try? model.prediction(conv1d_input: input) else {
//            return
//        }
//        let score = output.Identity[0].floatValue
//        DispatchQueue.main.async {
//            if score > 0.8 {
//                self.reps += 1
//                self.feedback = "Good squat!"
//            } else {
//                self.feedback = "Bad form!"
//            }
//        }
//    }
//
//    private func createMLMultiArray(from sequence: [[Float]]) -> MLMultiArray? {
//        guard let array = try? MLMultiArray(shape: [1, 30, 48], dataType: .float32) else { return nil }
//        for i in 0..<30 {
//            for j in 0..<48 {
//                array[[0, NSNumber(value: i), NSNumber(value: j)]] = NSNumber(value: sequence[i][j])
//            }
//        }
//        return array
//    }
//
//    private func resample(sequence: [[Float]], to targetLength: Int) -> [[Float]] {
//        let currentLen = sequence.count
//        guard currentLen > 0 else {
//            return Array(repeating: [Float](repeating: 0, count: 48), count: targetLength)
//        }
//        let step = Double(currentLen - 1) / Double(targetLength - 1)
//        return (0..<targetLength).map { i in
//            sequence[min(Int(round(Double(i) * step)), currentLen - 1)]
//        }
//    }
//}
