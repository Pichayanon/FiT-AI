import SwiftUI

struct WorkoutDetailView: View {
    let setTitle: String
    
    let exercises: [WorkoutExercise] = [
        WorkoutExercise(name: "Squat", imageName: "squat", reps: "15 reps"),
        WorkoutExercise(name: "High Knees", imageName: "highknees", reps: "10 reps"),
        WorkoutExercise(name: "Mountain Climbers", imageName: "mountain", reps: "20 reps")
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(setTitle)
                .font(.largeTitle)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .padding(.top)

            Text("This set includes:")
                .font(.headline)
                .foregroundColor(.gray)

            ScrollView {
                VStack(spacing: 16) {
                    ForEach(exercises) { exercise in
                        ExerciseCard(exercise: exercise)
                    }
                }
            }
            
            Spacer()
            
//            NavigationLink(destination: CameraWorkoutView()) {
//                Text("Start Workout")
//                    .frame(maxWidth: .infinity)
//                    .padding()
//                    .background(Color.yellow)
//                    .foregroundColor(.black)
//                    .cornerRadius(20)
//            }
//            .padding(.bottom, 20)
            NavigationLink(destination: WorkoutSessionView(setTitle: setTitle)) {
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

struct WorkoutExercise: Identifiable {
    let id = UUID()
    let name: String
    let imageName: String
    let reps: String
}

struct ExerciseCard: View {
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
