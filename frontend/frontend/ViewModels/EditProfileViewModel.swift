import Foundation
import SwiftUI

@MainActor
final class EditProfileViewModel: ObservableObject {
    @Published var name: String = "Pichayanon"
    @Published var age: String = "21"
    @Published var weight: String = "70"
    @Published var height: String = "175"

    func save() {
        print("Saved: \(name), \(age), \(weight), \(height)")
        // TODO: Persist to UserDefaults / backend when ready.
    }
}

