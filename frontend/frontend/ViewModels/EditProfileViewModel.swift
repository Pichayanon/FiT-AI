import Foundation
import SwiftUI

/// Manages form state for the edit profile screen and handles save operations.
///
/// This ViewModel is intentionally free of service dependencies so it can be
/// created as a @StateObject without needing @EnvironmentObject values at init
/// time. The caller passes the service explicitly to `save(using:)`.
@MainActor
final class EditProfileViewModel: ObservableObject {
    @Published var name: String = ""
    @Published var age: String = ""
    @Published var weight: String = ""
    @Published var height: String = ""

    init() {}

    /// Populates form fields from an existing profile, or clears them if nil.
    func fillFromProfile(_ profile: UserProfile?) {
        guard let p = profile else {
            name = ""
            age = ""
            weight = ""
            height = ""
            return
        }
        name = p.name
        age = p.age
        weight = p.weight
        height = p.height
    }

    /// Saves the current form values to Firestore via the provided profile service.
    func save(using profileService: ProfileService) async {
        let profile = UserProfile(name: name, age: age, weight: weight, height: height)
        await profileService.save(profile)
    }
}
