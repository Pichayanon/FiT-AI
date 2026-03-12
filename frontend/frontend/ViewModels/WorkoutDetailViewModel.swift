import Foundation
import SwiftUI

/// Provides workout program data for the workout detail/preview screen.
@MainActor
final class WorkoutDetailViewModel: ObservableObject {
    let program: WorkoutProgramDefinition

    var setTitle: String { program.title }
    var exercises: [WorkoutExercise] { program.exercises }

    init(setTitle: String) {
        self.program = WorkoutCatalog.program(for: setTitle)
    }
}
