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
    @State private var timer: Timer?

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ZStack {
                    CameraPreviewView(session: cameraManager.session)
                        .frame(height: UIScreen.main.bounds.height * 0.5)
                        .clipped()
                        .cornerRadius(12)
                        .overlay(
                            Text("Camera Live Preview")
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

                    Spacer()
                        .frame(height: 36)
                }
                .padding()
            }
            .background(Color.black.edgesIgnoringSafeArea(.all))
            .navigationTitle("Workout")
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

    func startSession() {
        isSessionRunning = true
        cameraManager.resetSequence()
        timer = Timer.scheduledTimer(withTimeInterval: 3.0, repeats: true) { _ in
            sendToAPI(sequence: cameraManager.keypointSequence)
            cameraManager.resetSequence()
        }
    }

    func stopSession() {
        isSessionRunning = false
        timer?.invalidate()
        timer = nil
    }

    func sendToAPI(sequence: [[Float]]) {
        guard let url = URL(string: "http://127.0.0.1:5050/predict") else {
            print("Invalid URL")
            return
        }

        let payload = ["sequence": sequence]

        guard let jsonData = try? JSONSerialization.data(withJSONObject: payload, options: []) else {
            print("Failed to encode sequence")
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Request failed:", error)
                return
            }

            guard let data = data else {
                print("No data received")
                return
            }

            if let result = try? JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
               let prediction = result["prediction"] as? Int,
               let confidence = result["confidence"] as? Float {

                DispatchQueue.main.async {
                    totalReps += 1
                    if prediction == 1 {
                        correctReps += 1
                        feedback = "Good form! Confidence: \(String(format: "%.2f", confidence))"
                    } else {
                        incorrectReps += 1
                        feedback = "Try adjusting posture. Confidence: \(String(format: "%.2f", confidence))"
                    }
                }
            } else {
                print("Failed to parse response")
            }
        }.resume()
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
