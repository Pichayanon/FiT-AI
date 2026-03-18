import SwiftUI
import FirebaseCore

/// Root application entry point. Configures Firebase and routes between Login and Home
/// based on the current authentication state.
///
/// Both `authService` and `profileService` are injected as environment objects so that
/// any descendant view can access them without prop-drilling through intermediate views.
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
                    HomeView()
                }
            }
            .environmentObject(authService)
            .environmentObject(profileService)
        }
    }
}
