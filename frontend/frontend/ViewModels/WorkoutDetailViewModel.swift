import Foundation
import SwiftUI

@MainActor
final class WorkoutDetailViewModel: ObservableObject {
    let setTitle: String
    let exercises: [WorkoutExercise]

    init(setTitle: String) {
        self.setTitle = setTitle
        self.exercises = [
            WorkoutExercise(name: "Wall-Sit", imageName: "wallsit", reps: "5s hold"),
            WorkoutExercise(name: "Squat", imageName: "squat", reps: "3 correct reps"),
            WorkoutExercise(name: "Plank", imageName: "plank", reps: "5s hold")
        ]
    }
}

