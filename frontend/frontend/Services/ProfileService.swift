import Foundation
import SwiftUI
import FirebaseAuth
import FirebaseFirestore

/// Loads and saves the current user's profile from Firestore (collection: "users", document: uid).
///
/// Automatically loads the profile when the user signs in and clears it on sign-out
/// via a Firebase auth state listener.
@MainActor
final class ProfileService: ObservableObject {
    @Published private(set) var currentProfile: UserProfile?
    @Published private(set) var isSaving = false
    @Published var errorMessage: String?

    private let db = Firestore.firestore()
    private var authListener: AuthStateDidChangeListenerHandle?

    init() {
        authListener = Auth.auth().addStateDidChangeListener { [weak self] _, user in
            Task { @MainActor in
                if user != nil {
                    await self?.load()
                } else {
                    self?.currentProfile = nil
                }
            }
        }
        if Auth.auth().currentUser != nil {
            Task { await load() }
        }
    }

    deinit {
        if let handle = authListener {
            Auth.auth().removeStateDidChangeListener(handle)
        }
    }

    var uid: String? { Auth.auth().currentUser?.uid }

    /// Loads the user profile from Firestore. Falls back to an empty profile on error.
    func load() async {
        guard let uid = uid else { currentProfile = nil; return }
        errorMessage = nil
        do {
            let doc = try await db.collection("users").document(uid).getDocument()
            if let data = doc.data() {
                currentProfile = UserProfile.from(data)
            } else {
                currentProfile = UserProfile.empty
            }
        } catch {
            errorMessage = error.localizedDescription
            currentProfile = UserProfile.empty
        }
    }

    /// Saves the given profile to Firestore, merging with existing data.
    func save(_ profile: UserProfile) async {
        guard let uid = uid else {
            errorMessage = "Not logged in"
            return
        }
        isSaving = true
        errorMessage = nil
        defer { isSaving = false }
        do {
            try await db.collection("users").document(uid).setData(profile.dictionary, merge: true)
            currentProfile = profile
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Clears any displayed error message.
    func clearError() {
        errorMessage = nil
    }
}
