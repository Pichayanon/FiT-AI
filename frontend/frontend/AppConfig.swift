import Foundation

enum AppConfig {
    private enum Key {
        static let wallSitWebSocketURL = "WALL_SIT_WS_URL"
        static let squatWebSocketURL = "SQUAT_WS_URL"
        static let plankWebSocketURL = "PLANK_WS_URL"
        static let lungesWebSocketURL = "LUNGES_WS_URL"
    }

    static func webSocketURL(for mode: WorkoutSessionMode) -> String {
        switch mode {
        case .wallSit:
            return requiredString(for: Key.wallSitWebSocketURL)
        case .squat:
            return requiredString(for: Key.squatWebSocketURL)
        case .plank:
            return requiredString(for: Key.plankWebSocketURL)
        case .lunges:
            return requiredString(for: Key.lungesWebSocketURL)
        }
    }

    private static func requiredString(for key: String) -> String {
        guard let value = Bundle.main.object(forInfoDictionaryKey: key) as? String,
              !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            fatalError("Missing app config value for \(key)")
        }
        return value
    }
}
