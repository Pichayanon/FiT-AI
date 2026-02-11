import Foundation
import SwiftUI

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

    /// โหลดค่าจากโปรไฟล์ (เรียกเมื่อเปิดหน้าหรือเมื่อ profile เปลี่ยน)
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

    /// บันทึกลง Firestore
    func save() async {
        let profile = UserProfile(name: name, age: age, weight: weight, height: height)
        await profileService.save(profile)
    }
}
