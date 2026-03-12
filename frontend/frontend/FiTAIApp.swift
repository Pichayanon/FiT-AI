import SwiftUI
import FirebaseCore

/// Root application entry point. Configures Firebase and routes between Login and Home
/// based on the current authentication state.
@main
struct FiTAIApp: App {
    @StateObject private var authService = AuthService()
    @StateObject private var profileService = ProfileService()

    init() {
        FirebaseApp.configure()
    }

    var body: some Scene {
        WindowGroup {
            Group {
                if authService.currentUser == nil {
                    LoginView(authService: authService)
                } else {
                    HomeView(authService: authService, profileService: profileService)
                }
            }
        }
    }
}
