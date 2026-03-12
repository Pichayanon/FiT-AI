import Foundation
import SwiftUI

@MainActor
final class WorkoutDetailViewModel: ObservableObject {
    let program: WorkoutProgramDefinition

    var setTitle: String { program.title }
    var exercises: [WorkoutExercise] { program.exercises }

    init(setTitle: String) {
        self.program = WorkoutCatalog.program(for: setTitle)
    }
}
