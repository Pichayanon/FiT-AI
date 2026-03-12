import Foundation
import SwiftUI

@MainActor
final class HomeViewModel: ObservableObject {
    @Published var userName: String = "Pichayanon"
    let programs = WorkoutCatalog.programs
}
