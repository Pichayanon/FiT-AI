import SwiftUI
import FirebaseCore

@main
struct frontendApp: App {
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
