import SwiftUI

struct WorkoutResultView: View {
    @StateObject private var viewModel: WorkoutResultViewModel
    
    @Environment(\.presentationMode) var presentationMode

    init(totalReps: Int, correctReps: Int, incorrectReps: Int, totalTime: Int, estimatedCalories: Int) {
        _viewModel = StateObject(
            wrappedValue: WorkoutResultViewModel(
                totalReps: totalReps,
                correctReps: correctReps,
                incorrectReps: incorrectReps,
                totalTime: totalTime,
                estimatedCalories: estimatedCalories
            )
        )
    }
    
    var body: some View {
        VStack(spacing: 20) {
            Text("Workout Summary")
                .font(.title2)
                .foregroundColor(.white)

            VStack(spacing: 12) {
                ResultStat(label: "Calories Burned", value: "\(viewModel.estimatedCalories) kcal")
                ResultStat(label: "Time Spent", value: viewModel.formattedTime)
                ResultStat(label: "Correct Reps", value: "\(viewModel.correctReps)")
                ResultStat(label: "Incorrect Reps", value: "\(viewModel.incorrectReps)")
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

fileprivate struct ResultStat: View {
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
