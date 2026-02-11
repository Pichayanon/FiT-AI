import Foundation
import FirebaseFirestore

/// โปรไฟล์ผู้ใช้ที่เก็บใน Firestore (collection: users, document: uid)
struct UserProfile {
    var name: String
    var age: String
    var weight: String
    var height: String

    static let empty = UserProfile(name: "", age: "", weight: "", height: "")

    var dictionary: [String: Any] {
        [
            "name": name,
            "age": age,
            "weight": weight,
            "height": height,
            "updatedAt": FieldValue.serverTimestamp()
        ]
    }

    static func from(_ data: [String: Any]) -> UserProfile {
        UserProfile(
            name: data["name"] as? String ?? "",
            age: data["age"] as? String ?? "",
            weight: data["weight"] as? String ?? "",
            height: data["height"] as? String ?? ""
        )
    }
}
