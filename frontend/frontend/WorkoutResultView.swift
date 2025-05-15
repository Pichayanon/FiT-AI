import SwiftUI

struct WorkoutResultView: View {
    let totalReps: Int
    let correctReps: Int
    let incorrectReps: Int
    let totalTime: Int
    let estimatedCalories: Int
    
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Workout Summary")
                .font(.title2)
                .foregroundColor(.white)

            VStack(spacing: 12) {
                ResultStat(label: "Calories Burned", value: "\(estimatedCalories) kcal")
                ResultStat(label: "Time Spent", value: "\(totalTime / 60) min \(totalTime % 60) sec")
                ResultStat(label: "Correct Reps", value: "\(correctReps)")
                ResultStat(label: "Incorrect Reps", value: "\(incorrectReps)")
            }

            Spacer()
            Button("Back to Home") {
                presentationMode.wrappedValue.dismiss()
            }
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.yellow)
            .foregroundColor(.black)
            .cornerRadius(12)
        }
        .padding()
        .background(Color.black.edgesIgnoringSafeArea(.all))
    }
}

struct ResultStat: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.gray)
            Spacer()
            Text(value)
                .foregroundColor(.white)
        }
        .font(.body)
    }
}
