import Foundation
import FirebaseFirestore

/// User profile stored in Firestore (collection: "users", document: uid).
/// Contains basic personal information used for display and calorie estimation.
struct UserProfile {
    var name: String
    var age: String
    var weight: String
    var height: String

    /// A blank profile used as the default before data is loaded.
    static let empty = UserProfile(name: "", age: "", weight: "", height: "")

    /// Firestore-compatible dictionary representation.
    var dictionary: [String: Any] {
        [
            "name": name,
            "age": age,
            "weight": weight,
            "height": height,
            "updatedAt": FieldValue.serverTimestamp()
        ]
    }

    /// Creates a `UserProfile` from a raw Firestore document dictionary.
    static func from(_ data: [String: Any]) -> UserProfile {
        UserProfile(
            name: data["name"] as? String ?? "",
            age: data["age"] as? String ?? "",
            weight: data["weight"] as? String ?? "",
            height: data["height"] as? String ?? ""
        )
    }
}
