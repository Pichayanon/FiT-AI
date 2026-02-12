import Foundation
import FirebaseAuth
import FirebaseFirestore

/// Saves completed workout sessions to Firestore (collection: workoutSessions).
/// Each document has: userId, setTitle, completedAtSeconds, totalTimeSeconds, estimatedCalories, exercises[].
/// Firestore rules: allow create when request.auth.uid == resource.data.userId.
@MainActor
final class WorkoutHistoryService {
    private let db = Firestore.firestore()

    var currentUserId: String? { Auth.auth().currentUser?.uid }

    /// Save a completed session for the current user. No-op if not logged in.
    func saveSession(summary: SessionSummary, setTitle: String) async {
        guard let uid = currentUserId else { return }
        let record = summary.toRecord(userId: uid, setTitle: setTitle)
        do {
            let data = try JSONEncoder().encode(record)
            guard let dict = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            try await db.collection("workoutSessions").addDocument(data: dict)
        } catch {
            print("[WorkoutHistory] save error: \(error)")
        }
    }

    /// Fetch recent sessions for the current user.
    func fetchSessions(limit: Int = 50) async throws -> [(id: String, record: WorkoutSessionRecord)] {
        guard let uid = currentUserId else { return [] }
        let snapshot = try await db.collection("workoutSessions")
            .whereField("userId", isEqualTo: uid)
            .order(by: "completedAtSeconds", descending: true)
            .limit(to: limit)
            .getDocuments()
        return snapshot.documents.compactMap { doc in
            guard let record = WorkoutSessionRecord(from: doc.data()) else { return nil }
            return (id: doc.documentID, record: record)
        }
    }
}
