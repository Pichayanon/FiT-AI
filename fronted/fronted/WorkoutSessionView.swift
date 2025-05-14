import SwiftUI

struct WorkoutSessionView: View {
    let setTitle: String
    
    @State private var feedback: String = "Try bending your knees less"
    @State private var totalReps = 10
    @State private var correctReps = 7
    @State private var incorrectReps = 3
    @State private var startTime = Date()
    @State private var navigateToResult = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Camera Placeholder
                ZStack {
                    Color.gray.opacity(0.3)
                    Text("Camera Preview Placeholder")
                        .foregroundColor(.white)
                        .font(.caption)
                }
                .frame(height: UIScreen.main.bounds.height * 0.5)
                
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

                    Spacer()

                    Button("Finish Workout") {
                        navigateToResult = true
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(12)
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
            }
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
