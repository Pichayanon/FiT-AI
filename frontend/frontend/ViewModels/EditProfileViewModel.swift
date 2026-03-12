import Foundation
import SwiftUI

/// Manages form state for the edit profile screen and handles save operations.
@MainActor
final class EditProfileViewModel: ObservableObject {
    @Published var name: String = ""
    @Published var age: String = ""
    @Published var weight: String = ""
    @Published var height: String = ""

    private let profileService: ProfileService

    init(profileService: ProfileService) {
        self.profileService = profileService
        fillFromProfile(profileService.currentProfile)
    }

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

    /// Saves the current form values to Firestore via the profile service.
    func save() async {
        let profile = UserProfile(name: name, age: age, weight: weight, height: height)
        await profileService.save(profile)
    }
}
