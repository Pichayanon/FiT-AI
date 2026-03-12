import Foundation
import SwiftUI

/// Provides data for the home screen, including the list of available workout programs.
@MainActor
final class HomeViewModel: ObservableObject {
    @Published var userName: String = "Guest User"
    let programs = WorkoutCatalog.programs
}
