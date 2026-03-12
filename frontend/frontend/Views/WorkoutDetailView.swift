import SwiftUI

/// Workout detail/preview screen showing the exercise list and a Start Workout button.
struct WorkoutDetailView: View {
    @StateObject private var viewModel: WorkoutDetailViewModel

    init(setTitle: String) {
        _viewModel = StateObject(wrappedValue: WorkoutDetailViewModel(setTitle: setTitle))
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(viewModel.setTitle)
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
                .allowsTightening(true)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top)

            Text("This set includes:")
                .font(.headline)
                .foregroundColor(.gray)

            ScrollView {
                VStack(spacing: 16) {
                    ForEach(viewModel.exercises) { exercise in
                        ExerciseCard(exercise: exercise)
                    }
                }
            }
            
            Spacer()
            
            NavigationLink(destination: WorkoutSessionView(setTitle: viewModel.setTitle)) {
                Text("Start Workout")
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.yellow)
                    .foregroundColor(.black)
                    .cornerRadius(20)
            }
            .padding(.bottom, 20)
        }
        .padding(.horizontal)
        .background(Color.black.edgesIgnoringSafeArea(.all))
    }
}

// MARK: - Exercise Card

/// Card showing an exercise image, name, and rep/hold target.
fileprivate struct ExerciseCard: View {
    let exercise: WorkoutExercise

    var body: some View {
        HStack(spacing: 16) {
            Image(exercise.imageName)
                .resizable()
                .aspectRatio(contentMode: .fill)
                .frame(width: 100, height: 80)
                .cornerRadius(12)
                .clipped()

            VStack(alignment: .leading, spacing: 4) {
                Text(exercise.name)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                Text(exercise.reps)
                    .font(.subheadline)
                    .foregroundColor(.gray)
            }

            Spacer()
        }
        .padding()
        .background(Color.gray.opacity(0.15))
        .cornerRadius(16)
    }
}
