import SwiftUI

struct WorkoutSessionView: View {
    let setTitle: String

    @ObservedObject var cameraManager = CameraManager()

    @State private var feedback: String = "Waiting for result..."
    @State private var totalReps = 0
    @State private var correctReps = 0
    @State private var incorrectReps = 0
    @State private var startTime = Date()
    @State private var navigateToResult = false
    @State private var isSessionRunning = false

    private let wsURL = "ws://IP:5050/ws/video"
    
    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ZStack {
                    CameraPreviewView(session: cameraManager.session)
                        .frame(height: UIScreen.main.bounds.height * 0.5)
                        .clipped()
                        .cornerRadius(12)
                        .overlay(
                            Text(isSessionRunning ? "Streaming..." : "Camera Live Preview")
                                .foregroundColor(.white)
                                .padding(6)
                                .background(Color.black.opacity(0.5))
                                .cornerRadius(8),
                            alignment: .topLeading
                        )
                }

                VStack(spacing: 16) {
                    Text("Workout Set: \(setTitle)")
                        .foregroundColor(.yellow)
                        .font(.headline)

                    Text(feedback)
                        .font(.title3)
                        .foregroundColor(.white)
                        .padding()
                        .frame(maxWidth: .infinity)
                        .background(Color.black.opacity(0.6))
                        .cornerRadius(12)

                    HStack {
                        RepCard(title: "Total", value: "\(totalReps)")
                        RepCard(title: "Correct", value: "\(correctReps)", color: .green)
                        RepCard(title: "Incorrect", value: "\(incorrectReps)", color: .red)
                    }

                    if let err = cameraManager.lastError {
                        Text("Error: \(err)")
                            .foregroundColor(.red)
                            .font(.caption)
                    }

                    if isSessionRunning {
                        Button("End Session") {
                            stopSession()
                            navigateToResult = true
                        }
                        .padding()
                        .background(Color.red)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    } else {
                        Button("Start Session") {
                            startSession()
                        }
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }

                    Spacer().frame(height: 36)
                }
                .padding()
            }
            .background(Color.black.edgesIgnoringSafeArea(.all))
            .navigationTitle("Workout")
            .onAppear {
                cameraManager.startSession()

                // รับข้อความจาก backend แล้วอัปเดต UI
                cameraManager.onBackendMessage = { text in
                    handleBackendMessage(text)
                }
            }
            .onDisappear {
                stopSession()
                cameraManager.stopSession()
            }
            .navigationDestination(isPresented: $navigateToResult) {
                WorkoutResultView(
                    totalReps: totalReps,
                    correctReps: correctReps,
                    incorrectReps: incorrectReps,
                    totalTime: Int(Date().timeIntervalSince(startTime)),
                    estimatedCalories: totalReps * 4
                )
                .navigationBarBackButtonHidden(true)
            }
        }
    }

    private func startSession() {
        isSessionRunning = true
        startTime = Date()
        feedback = "Streaming to backend..."
        totalReps = 0
        correctReps = 0
        incorrectReps = 0

        cameraManager.startStreaming(to: wsURL)
    }

    private func stopSession() {
        isSessionRunning = false
        cameraManager.stopStreaming()
    }

    private func handleBackendMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        let type = obj["type"] as? String ?? ""
        if type == "result" {
            let fb = obj["feedback"] as? String ?? "..."
            let t = obj["totalReps"] as? Int ?? totalReps
            let c = obj["correctReps"] as? Int ?? correctReps
            let ic = obj["incorrectReps"] as? Int ?? incorrectReps

            DispatchQueue.main.async {
                feedback = fb
                totalReps = t
                correctReps = c
                incorrectReps = ic
            }
        } else if type == "info" {
            let msg = obj["message"] as? String ?? "..."
            DispatchQueue.main.async { feedback = msg }
        }
    }
}

struct RepCard: View {
    let title: String
    let value: String
    var color: Color = .white

    var body: some View {
        VStack {
            Text(value)
                .font(.title)
                .fontWeight(.bold)
                .foregroundColor(color)
            Text(title)
                .font(.caption)
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity)
    }
}
