import Foundation
import SwiftUI

@MainActor
final class WorkoutDetailViewModel: ObservableObject {
    let setTitle: String
    let exercises: [WorkoutExercise]

    init(setTitle: String) {
        self.setTitle = setTitle
        self.exercises = [
            WorkoutExercise(name: "Squat", imageName: "squat", reps: "15 reps"),
            WorkoutExercise(name: "High Knees", imageName: "highknees", reps: "10 reps"),
            WorkoutExercise(name: "Mountain Climbers", imageName: "mountain", reps: "20 reps")
        ]
    }
}

